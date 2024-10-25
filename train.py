from ultralytics.models import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr, emojis, clean_url
import ultralytics.utils.callbacks.tensorboard as tb_module
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributed as dist
import warnings
from torch.nn.utils import spectral_norm
from torchvision.ops import roi_align, box_convert
from functools import partial
import random
import os



#torch.use_deterministic_algorithms(False)
def seed_everything(seed=9527):
    # 固定隨機種子, 藉此讓整體訓練過程可復現
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class GradientReversalFunction(torch.autograd.Function):
   #Funzione di inversione del gradiente, utilizzata per modificare la direzione del gradiente durante 
   # la retropropagazione del gradiente
   
    @staticmethod
    def forward(ctx, x, alpha):
        """
        La propagazione in avanti restituisce direttamente l'input x.
        Args:
            ctx: oggetto di contesto, utilizzato per salvare informazioni da utilizzare durante la backpropagation
            x: tensore di ingresso
            alfa: coefficiente dello strato di inversione del gradiente, che controlla la forza dell'inversione del gradiente
        Returns:
            un tensore con la stessa forma dell'input x
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        La backpropagation moltiplica il gradiente in entrata per -alfa per ottenere l'inversione del gradiente
        Args:
            ctx: informazioni sul contesto salvate
            grad_output: propaga qui il gradiente all'indietro
        Returns:
           Il gradiente di output viene moltiplicato per -alpha e restituisce None 
           per alpha poiché alpha viene solitamente trattato come una costante
        """
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha=1.):
        """
        Inizializza il livello di inversione del gradiente.
        Args:
            alpha: coefficiente di inversione del gradiente, il valore predefinito è 1,0
        """
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        模組的前向傳播邏輯, 透過使用 GradientReversalFunction 實現梯度反轉。
        Args:
            x: tensore di ingresso
        Returns:
            Tensore elaborato dalla funzione di inversione del gradiente
        """
        return GradientReversalFunction.apply(x, self.alpha)


class SpectralLinear(nn.Linear):
    # 使用頻譜歸一化的線性層
    def __init__(self, *args, **kwargs):
        """
        Inizializza uno strato lineare di normalizzazione spettrale, ereditato da nn.Linear, e aggiunge la normalizzazione spettrale ai pesi.
        """
        super().__init__(*args, **kwargs)
        spectral_norm(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class SpectralConv2d(nn.Conv2d):
    # 使用頻譜歸一化的二維卷積層
    def __init__(self, *args, **kwargs):
        """
        Inizializza uno strato convoluzionale 2D normalizzato spettralmente, ereditato da nn.Conv2d, e aggiungi la normalizzazione spettrale ai pesi.
        """
        super().__init__(*args, **kwargs)
        spectral_norm(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class DiscriminatorHead(nn.Module):
    def __init__(self, dim_in, dim_h, dim_o=1):
        """
        Inizializza la struttura della testa del discriminatore, inclusa una sequenza di più livelli.
        Args:
            dim_in: La dimensione della funzione di input
            dim_h: dimensione del livello nascosto
            dim_o: dimensione del livello di output, il valore predefinito è 1
        """
        super().__init__()
        # 將輸入特徵展平 並通過一個線性層轉換到隱藏層維度
        self.to_flat = nn.Sequential(
            SpectralConv2d(dim_in, dim_h // 2, kernel_size=1),
            nn.Flatten(),
            nn.LazyLinear(dim_h),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 隱藏層的多個線性層
        self.neck = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_h // 2, dim_h // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ) for _ in range(3)
        ])
        # 最終的線性層
        self.head = nn.Sequential(
            SpectralLinear(dim_h // 2 * 4, dim_h // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim_h // 2, dim_o, bias=False),
        )

    def forward(self, x):
        # 通過一個線性層將輸入特徵展平
        x = self.to_flat(x)
        # 類似於 CSPNet 的結構，將展平後的特徵分成多份, 並最終聚合不同深度的特徵
        x = x.split(x.shape[1] // 2, dim=1)
        xs = [x[0]]
        for m in self.neck:
            x = m(x[1]) if isinstance(x, tuple) else m(x)
            xs.append(x)
        # 聚合不同深度的特徵
        x = torch.cat(xs, dim=1)
        return self.head(x)


class Discriminator(nn.Module):
    def __init__(self, chs=None, amp=False):
        """
        Inizializza la struttura del discriminatore, inclusi lo strato di inversione del gradiente e vari strati convoluzionali.
        Args:
            chs: Lista del numero di canali, definisce il numero di canali di input e output per ogni strato convoluzionale
            amp: Se utilizzare l'addestramento con precisione mista automatica, impostato su False di default
        """
        super().__init__()
        # Qui si corrisponde al numero di canali degli strati collegati nella struttura YOLO
        if chs is None:
            chs = [64, 128, 256]
            self.chs = chs
            self.f_len = len(chs)
        # Inizializza lo strato di inversione del gradiente
        self.grl = GradientReversalLayer(alpha=1.0)
        self.amp = amp
        # Strati convoluzionali per estrarre caratteristiche a diverse profondità, il numero di canali è definito da chs
        self.p = nn.ModuleList([
            nn.Sequential(
                # Primo strato convoluzionale, estrae caratteristiche utilizzando un kernel di convoluzione 11x11
                nn.Conv2d(chs[i] if i == 0 else chs[i] * 2, 64, kernel_size=11, stride=2, padding=5, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                # Secondo strato convoluzionale, utilizza un kernel di convoluzione 1x1 per la trasformazione dimensionale delle caratteristiche
                nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                # Terzo strato convoluzionale, utilizza un kernel di convoluzione 1x1 per la trasformazione dimensionale delle caratteristiche, 
                # con l'obiettivo di raggiungere il numero di canali del livello successivo di chs
                nn.Conv2d(32, chs[i + 1] if i + 1 < len(chs) else chs[i], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(chs[i + 1] if i + 1 < len(chs) else chs[i]),
                nn.SiLU(inplace=True),
            ) for i in range(len(chs))
        ])
        
        # Struttura principale dopo aver aggregato tutte le caratteristiche di chs
        self.head = DiscriminatorHead(chs[-1], 256)
        # Ottimizzatore separato dal modello del detector, utilizzato solo per addestrare il discriminatore
        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, fs: list[torch.tensor]):
        with torch.cuda.amp.autocast(self.amp):
            assert len(fs) == self.f_len, f'Attesi {self.f_len} mappe di caratteristiche, ottenute {len(fs)}'
            # Dopo aver ricevuto le caratteristiche dal modello del detector, si invertono i gradienti attraverso lo strato di inversione del gradiente
            # In questo modo, il modello del detector funge da generatore, mentre il discriminatore effettua una "discriminazione di dominio" sulle caratteristiche del modello del detector
            fs = [self.grl(f) for f in fs]
            # Struttura per l'aggregazione delle caratteristiche attraverso risoluzioni multiple
            x = self.p[0](fs[0])
            for i in range(1, len(fs)):
                x = torch.cat((x, fs[i]), dim=1)
                x = self.p[i](x)
            # Alla fine, la previsione viene fatta attraverso la struttura principale
            return self.head(x)


class CustomTrainer(DetectionTrainer):
    """
    Questo è un trainer personalizzato, che consente l'allenamento simultaneo di un dominio di origine (source domain) e di un dominio di destinazione (target domain).
    Il ciclo di allenamento si basa su ultralytics.models.yolo.detect.train.DetectionTrainer ed è stato esteso,
    ma durante l'allenamento aggiunge logiche relative all'adattamento del dominio (domain adaptation) e all'allenamento del discriminatore.
    """
    def __init__(self, target_domain_data_cfg, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        # Caricamento aggiuntivo del dataset per il dominio di destinazione
        try:
            if target_domain_data_cfg in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.t_data = check_det_dataset(target_domain_data_cfg)
        except Exception as e:
            raise RuntimeError(emojis(f"Errore nel dataset '{clean_url(target_domain_data_cfg)}' ❌ {e}")) from e
        # Inizializzazione del dataset e dataloader del dominio di destinazione
        self.t_trainset, self.t_testset = self.get_dataset()
        self.t_iter = None
        self.t_train_loader = None
        # Inizializzazione dei parametri relativi all'adattamento del dominio e degli hook del modello
        self.model_hook_handler = []
        # Impostazione degli indici degli strati (layer) da agganciare (hook), qui si agganciano 3 strati nel backbone di YOLO
        self.model_hook_layer_idx: list[int] = [2, 4, 6]
        # Dimensioni ROI corrispondenti alle diverse dimensioni delle mappe di caratteristiche
        self.roi_size = list(reversed([20 * 2 ** i for i in range(len(self.model_hook_layer_idx))]))
        # Inizializzazione delle caratteristiche agganciate dal hook
        self.model_hooked_features: None | list[torch.tensor] = None
        # Inizializzazione del modello discriminatore e delle teste di proiezione
        self.discriminator_model = None
        self.projection_model = None
        self.additional_models = []
        # Il metodo on_train_start è un callback eseguito durante l'allenamento di DetectionTrainer e viene utilizzato per inizializzare i modelli helper
        self.add_callback('on_train_start', self.init_helper_model)

    def init_helper_model(self, *args, **kwargs):
        # Poiché la variabile self.amp viene inizializzata nel corso dell'allenamento di DetectionTrainer e il Discriminatore ha bisogno di usarla,
        # dobbiamo inizializzare il Discriminatore prima che l'intero trainer e il modello siano completamente inizializzati e prima dell'inizio dell'allenamento, in modo che utilizzi la stessa configurazione.
        self.discriminator_model = Discriminator(amp=self.amp).to(self.device)
        # Aggiunta del modello discriminatore alla lista dei modelli aggiuntivi, in modo che possa essere ottimizzato insieme agli altri durante optimizer_step
        #in un unica passata di addestramento.
        self.additional_models.append(self.discriminator_model)

    def get_t_batch(self):
        # Ottenimento di un batch dal dominio di destinazione
        #Verifica se iteratore inizializzato altrimenti si procede a crearlo
        if self.t_iter is None:
            # Dato che ad ogni chiamata si preleva solo un batch, convertiamo il dataloader in un iteratore per facilitare l'iterazione
            self.t_train_loader = self.get_dataloader(self.t_trainset, batch_size=self.batch_size, rank=RANK, mode='train')
            self.t_iter = iter(self.t_train_loader)
        try:
            # Il metodo next lancia un errore StopIteration quando termina l'iterazione, quindi lo gestiamo con try-except
            batch = next(self.t_iter)
        except StopIteration:
            # Quando l'iterazione finisce, reinizializziamo il dataloader e lo riconvertiamo in un iteratore
            self.t_iter = iter(self.t_train_loader)
            batch = next(self.t_iter)
        return batch   #ritorno del batch

    def activate_hook(self, layer_indices: list[int] = None):
        # Attivazione del hook, che consente di ottenere le caratteristiche dai layer specificati
        if layer_indices is not None:
            self.model_hook_layer_idx = layer_indices
        self.model_hooked_features = [None for _ in self.model_hook_layer_idx]
        self.model_hook_handler = \
            [self.model.model[l].register_forward_hook(self.hook_fn(i)) for i, l in enumerate(self.model_hook_layer_idx)]   #Gli hook vengono registrati utilizzando register_forward_hook, 
                                                                                                                            #che permette di eseguire una funzione (hook_fn) ogni volta che i dati passano attraverso il layer
    def deactivate_hook(self):
        # Disattivazione del hook e rimozione di tutti gli hook
        if self.model_hook_handler is not None:
            for hook in self.model_hook_handler:
                hook.remove()
            self.model_hooked_features = None      #feautures agganciate vengono impostate su None
            self.model_hook_handler = []

    def hook_fn(self, hook_idx):
        # Definizione della funzione hook e salvataggio nella lista degli hook all'indice specificato
        def hook(m, i, o):
            self.model_hooked_features[hook_idx] = o
        return hook

    def get_dis_output_from_hooked_features(self, batch):
        # Passa le caratteristiche agganciate (hooked) al modello discriminatore per la previsione
        if self.model_hooked_features is not None:
            # Trasformazione delle dimensioni e delle coordinate dei dati
            bbox_batch_idx = batch['batch_idx'].unsqueeze(-1)
            bbox = batch['bboxes']
            bbox = box_convert(bbox, 'cxcywh', 'xyxy')
            # Ridimensionamento dei bbox e utilizzo di ROI Align per l'estrazione delle caratteristiche
            rois = []
            for fidx, f in enumerate(self.model_hooked_features):
                f_bbox = bbox * f.shape[-1]
                f_bbox = torch.cat((bbox_batch_idx, f_bbox), dim=-1)
                f_roi = roi_align(f, f_bbox.to(f.device), output_size=self.roi_size[fidx], aligned=True)
                rois.append(f_roi)
            # Aggregazione delle caratteristiche di diverse risoluzioni e previsione tramite il modello discriminatore
            dis_output = self.discriminator_model(rois)
            return dis_output
        else:
            return None
    
    def optimizer_step(self, optims: None | list[torch.optim.Optimizer] = None):
        """Esegui un singolo step dell'ottimizzatore durante l'allenamento con clipping dei gradienti e aggiornamento EMA."""
        # Amplifica i gradienti dell'ottimizzatore per supportare il training con precisione mista
        self.scaler.unscale_(self.optimizer)
        # Se viene passato un altro ottimizzatore, amplifica anche i suoi gradienti (qui dovrebbe essere l'ottimizzatore del modello discriminatore)
        if optims is not None:
            for o in optims:
                # controlla se l'ottimizzatore ha gradienti
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.unscale_(o)
        # Taglia i gradienti troppo grandi
        max_norm = 10.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)  # clip dei gradienti
        if len(self.additional_models) > 0:
            for m in self.additional_models:
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=max_norm * 2)
        # Aggiorna l'ottimizzatore tramite lo scaler
        self.scaler.step(self.optimizer)
        if optims is not None:
            for o in optims:
                # controlla se l'ottimizzatore ha gradienti
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.step(o)
        # Aggiorna lo scaler
        self.scaler.update()
        # Reimposta i gradienti dell'ottimizzatore
        self.optimizer.zero_grad()
        if optims is not None:
            for o in optims:
                o.zero_grad()
        # Se è presente EMA, aggiorna l'EMA
        if self.ema:
            self.ema.update(self.model)

    def _do_train(self, world_size=1):
        # Regola le impostazioni del training distribuito
        #si verifica se l'allenamento è distribuito su più GPU e si configura di conseguenza.
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        # Impostazioni di alcuni parametri per il processo di training
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # numero di batch
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # iterazioni di warmup
        last_opt_step = -1

        # Qui viene chiamato il callback on_train_start, che corrisponde all'inizializzazione del modello discriminatore nel nostro codice
        self.run_callbacks('on_train_start')
        
        LOGGER.info(f'Dimensioni delle immagini {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Usando {self.train_loader.num_workers * (world_size or 1)} lavoratori del dataloader\n'
                    f"Risultati loggati in {colorstr('bold', self.save_dir)}\n"
                    f'Inizio del training per {self.epochs} epoche...')
        
        # Configurazione dei parametri durante il training
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb                   # ??? punto in cui chiudere funzionalità mosaic ???
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])             
        epoch = self.epochs  # predefinito per riprendere un modello completamente allenato        
        # Avvia il ciclo di training, attivando prima gli hook del modello              

        self.activate_hook()
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
                                                                                    
            # Aggiorna le impostazioni del dataloader
            #Disattiva la modalità mosaic raggiunta una certa epoca
            if epoch == (self.epochs - self.args.close_mosaic):                
                LOGGER.info('Chiusura del mosaic del dataloader')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()
            
            # Barra di progresso per il training distribuito
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            
            # Inizializzazione dei parametri durante il training, reset del loss e dell'ottimizzatore
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                # Richiama il callback on_train_batch_start (non è stato aggiunto nessun callback extra qui)
                self.run_callbacks('on_train_batch_start')
                # Warmup -> si regola il learning rate e altri parametri se numero iterazione fino a questo punto minore del numero di iterazioni impostate per il warmup
                ni = i + nb * epoch
                if ni <= nw:           
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # Forward pass
                with torch.cuda.amp.autocast(self.amp):
                    # Ottieni prima la loss dell'object detection per il dominio sorgente e le caratteristiche del forward hook
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    # Durante l'addestramento distribuito, moltiplica la loss per world_size per evitare che la loss sia troppo piccola
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items  #calcola la loss media su tutti i batch finora
                    # Ottieni l'output del discriminatore per il dominio sorgente
                    source_critics = self.get_dis_output_from_hooked_features(batch)
                    
                    # Successivamente, addestriamo il dominio target
                    t_batch = self.get_t_batch()
                    # Esegui un forward sul dominio target e ottieni l'output del discriminatore per il dominio target
                    t_batch = self.preprocess_batch(t_batch)
                    t_loss, t_loss_item = self.model(t_batch)
                    target_critics = self.get_dis_output_from_hooked_features(t_batch)
                    # Integra la loss dell'object detection del dominio sorgente e del dominio target
                    self.loss += t_loss

                    # Nei primi 6 epoch e negli ultimi 50 epoch non viene addestrato il discriminatore,
                    # all'inizio perché l'object detection non ha ancora caratteristiche rappresentative, quindi l'adattamento del dominio ha poco senso
                    # alla fine, per evitare che i gradienti instabili del GAN influenzino la performance dell'object detection
                    if 6 < epoch < self.args.epochs - 50:
                        # Calcola la loss del discriminatore, utilizzando la hinge loss
                        threshold = 20
                        loss_d = (F.relu(torch.ones_like(source_critics) * threshold + source_critics)).mean()  #loss discriminatore dominio sorgente
                        loss_d += (F.relu(torch.ones_like(target_critics) * threshold - target_critics)).mean() #Aggiunge loss discriminatore dominio target
                    else:
                        loss_d = 0
                    self.loss += loss_d * 2          # ??? Perchè moltiplica loss discriminatore per 2 ???   

                # Backpropagation
                self.scaler.scale(self.loss).backward()
                # Addestramento distribuito
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step(optims=[self.discriminator_model.optim])
                    last_opt_step = ni
                # Registra alcuni parametri dell'addestramento e fa il logging
                # Il codice successivo fino a deactive_hook è il processo di log e addestramento di ultralytics, quindi non ci sono ulteriori commenti
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    tb_module.WRITER.add_scalar('train/critic-source', source_critics.mean(), ni)
                    tb_module.WRITER.add_scalar('train/critic-target', target_critics.mean(), ni)
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

                self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # per i loggers

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')  # sopprime 'Detected lr_scheduler.step() before optimizer.step()'
                    self.scheduler.step()
                self.run_callbacks('on_train_epoch_end')

                if RANK in (-1, 0):
                    # Validazione
                    self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                    final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                    if self.args.val or final_epoch:
                        self.metrics, self.fitness = self.validate()
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                    self.stop = self.stopper(epoch + 1, self.fitness)

                    # Salva il modello
                    if self.args.save or (epoch + 1 == self.epochs):
                        # Nota: poiché il modello discriminatore non fa parte del modello detector, e non dovrebbero essere salvate caratteristiche del hook,
                        # è necessario disattivare il hook prima di salvare il modello
                        self.deactivate_hook()
                        self.save_model()
                        self.run_callbacks('on_model_save')
                        # Poiché l'addestramento continua, è necessario riattivare il hook
                        self.activate_hook()

                tnow = time.time()
                self.epoch_time = tnow - self.epoch_time_start
                self.epoch_time_start = tnow
                self.run_callbacks('on_fit_epoch_end')
                torch.cuda.empty_cache()  # svuota la vRAM della GPU alla fine dell'epoch, può aiutare con errori di memoria

                # Early Stopping
                if RANK != -1:  # se in addestramento DDP
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  #trasmette 'stop' a tutti i rank
                    if RANK != 0:
                        self.stop = broadcast_list[0]
                if self.stop:
                    break  # deve interrompere tutti i rank DDP

                if RANK in (-1, 0):
                    # Esegui la valutazione finale con best.pt
                    LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                                f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
                    self.final_eval()
                    if self.args.plots:
                        self.plot_metrics()
                    self.run_callbacks('on_train_end')

                # Dopo l'addestramento, disattiva il hook
                self.deactivate_hook()
                torch.cuda.empty_cache()
                self.run_callbacks('teardown')

def main():
    # 設定 Hyperparameters 和隨機種子
    seed = 95 * 27
    kwargs = {
        'imgsz': 640,
        'epochs': 100,
        'val': False,
        'workers': 2,
        'batch': 12,
        'seed': seed,
    }

    seed_everything(seed)
    
    # Carica modello preaddestrato
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # Inizializza trainer personalizzato
    custom_trainer = partial(CustomTrainer, target_domain_data_cfg='data/ACDC-rain.yaml')
    model.train(custom_trainer, data='data/ACDC-fog.yaml', name='train_RMD', patience=0, **kwargs)
    # rain domain
    model.val(data='data/ACDC-rain.yaml', name='val_RMD_rain')
    # fog domain 
    model.val(data='data/ACDC-fog.yaml', name='val_RMD_fog')

    # 在 fog domain 上進行訓練, 一樣先載入預訓練模型
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # 透過 ultryalitics 提供的預設訓練器進行訓練
    model.train(data='data/ACDC-fog.yaml', name='train_fog', **kwargs)
    # 將訓練好的模型在 fog domain 上進行測試
    model.val(data='data/ACDC-fog.yaml', name='val_fog_fog')
    # 將訓練好的模型在 rain domain 上進行測試
    model.val(data='data/ACDC-rain.yaml', name='val_fog_rain')

    # 在 rain domain 上進行訓練, 一樣先載入預訓練模型
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # 透過 ultryalitics 提供的預設訓練器進行訓練
    model.train(data='data/ACDC-rain.yaml', name='train_rain', **kwargs)
    # 將訓練好的模型在 rain domain 上進行測試
    model.val(data='data/ACDC-rain.yaml', name='val_rain_rain')
    # 將訓練好的模型在 fog domain 上進行測試
    model.val(data='data/ACDC-fog.yaml', name='val_rain_fog')

    # 在 rain 和 fog domain 上進行訓練, 一樣先載入預訓練模型
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # 透過 ultryalitics 提供的預設訓練器進行訓練
    model.train(data='data/ACDC-fog_rain.yaml', name='train_fog_rain-full-epoch', **kwargs)
    # 將訓練好的模型在 rain domain 上進行測試
    model.val(data='data/ACDC-rain.yaml', name='val_fograin_rain-full-epoch')
    # 將訓練好的模型在 fog domain 上進行測試
    model.val(data='data/ACDC-fog.yaml', name='val_fograin_fog-full-epoch')

if __name__ == '__main__':
    main()
