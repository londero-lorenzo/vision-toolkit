# Lavagna senza docente

Tool in Python per togliere il docente dalla lavagna nei video delle lezioni e sapere, istante per istante, cosa sta indicando.

# Perché esiste

Uno screenshot della lavagna piena, a fine lezione, perde il contesto: mentre spiega, il docente indica le parti di cui sta parlando, e quel legame serve per allineare la lavagna alla trascrizione nei riassunti.

Il tool produce:
- la lavagna ricostruita **senza docente**, con uno snapshot ogni volta che il contenuto cambia (scrittura o cancellazione)
- un log con timestamp del **punto indicato** dal docente, in coordinate della lavagna, da incrociare con i timestamp della trascrizione

# Come funziona

1. **Lavagna**: quadrilatero trovato sulla mediana temporale dei frame (il docente si muove e nella mediana sparisce), poi rettificato con un'omografia. In alternativa i 4 angoli si passano a mano (`--quad`), una volta per aula.
2. **Docente**: maschera da MediaPipe PoseLandmarker, più le braccia disegnate come segmenti tra i landmark spalla, gomito, polso, indice. Facoltativo: DeepLab V3 sul ritaglio della persona (`--refine`).
3. **Ricostruzione**: per ogni pixel della lavagna rettificata, mediana degli ultimi N campioni in cui non era coperto. Un pixel coperto ora era visibile poco prima o lo sarà poco dopo.
4. **Punto indicato**: punta dell'indice del braccio più sollevato rispetto al busto, portata in coordinate della lavagna.

# Togliere, non oscurare

Il docente si **toglie** dalla lavagna, non si oscura. Oscurarlo (blur, sagoma, colore medio) è peggio sotto entrambi gli aspetti:
- **privacy**: la sagoma resta e, a seconda del blur, la persona si riconosce ancora
- **contenuto**: il corpo copre proprio la lavagna che serve. Nel frame di prova (CIT, 11:33) la docente indica `H_2(P_1/2) = 1/2 log2 2 + 1/2 log2 2 = 1`, e il suo corpo copre la colonna con la definizione di `H_D(P)`

Il gesto serve, ma solo come dato: il punto indicato si estrae finché la persona è nel frame e si salva come metadato in `pointing.jsonl`, solo coordinate e niente pixel della persona. In uscita restano due prodotti separati, la lavagna pulita e il log del punto indicato. Sovrapponendoli si ottiene la lavagna pulita con la zona indicata in evidenza.

Ricostruire dai frame vicini risolve anche il caso in cui la mano copre proprio la formula indicata.

Nel vault di studio entrano solo gli snapshot puliti e `pointing.jsonl`. I frame con il docente, anche censurati, restano in `data/`, fuori da git.

# Dalla trascrizione alla formula

Il caso d'uso che guida il progetto: nella trascrizione si sente "Here we are calculating the entropy" a un certo minuto, e si vuole sapere quale formula sta indicando.

1. Si cerca la frase nella trascrizione e si legge il suo timestamp.
2. Si prende la finestra di frame attorno a quel timestamp.
3. Dentro la finestra si tengono solo i frame con un gesto di puntamento (`arm_angle`, persistenza su più campioni).
4. Si sovrappone la zona indicata allo snapshot pulito di quell'istante.
5. Claude legge la zona e dice quale formula è e come si lega a ciò che il docente sta dicendo.

**Granularità della trascrizione.** Le trascrizioni del vault (`*.audio.md`) hanno un timestamp per blocco di parlato (0:21, 1:27, 2:28…), non per frase. Da una frase si risale quindi a una finestra di circa un minuto, non al secondo. Ci sono due strade:
- dentro la finestra si scelgono i frame con un gesto di puntamento, come al passo 3 (basta per partire)
- oppure si ritrascrive l'audio con timestamp per parola (Whisper li fornisce), e la finestra scende a pochi secondi

**Precisione richiesta.** Basta una **zona** (un box sulla lavagna), non un punto preciso: la lettura fine della formula la fa Claude sul ritaglio. Questo conta soprattutto quando il docente indica da lontano, col braccio teso, invece di toccare la lavagna.

Difficoltà stimata, dalla più bassa:
- timestamp → frame (ffmpeg) e ricerca della frase nella trascrizione: facile
- lavagna pulita: media, è soprattutto fusione temporale; la finestra va tenuta corta perché la lavagna cambia (scrittura, cancellazione)
- puntamento da lontano: la parte più delicata

# Cosa è emerso dagli esperimenti

Su un frame della lezione di CIT (`notebooks/01_frame_singolo.ipynb`):

| Strada | Esito |
|---|---|
| DeepLab V3 sul frame intero (demo web MediaPipe) | 0 pixel di persona, confidenza massima 0.09 |
| DeepLab V3 su un ritaglio attorno al docente | persona trovata, confidenza 0.9997 |
| PoseLandmarker sul frame intero | persona trovata, indice sulla punta delle dita |
| Colore (pelle, ardesia, gesso in Lab) | ardesia compatta e separabile; pelle e gesso stessa luminosità, 10% dei pixel del braccio classificati come gesso |

- Il problema di DeepLab è la **risoluzione** (gira a 257×257, il docente diventa pochi pixel), non il colore. La media dei colori per regione aiutava ma presuppone la maschera: è circolare.
- Il colore distingue bene "non è lavagna", ma non la pelle o una camicia bianca dal gesso: utile solo per rifinire i bordi.
- Il passo decisivo è **temporale**: su un video sintetico (`tests/test_synthetic.py`) la ricostruzione ha 0% di pixel errati contro il 3,2% di un frame singolo con il docente davanti.

I numeri vengono da un frame in cui la docente è censurata con aree di colore medio: vanno riconfermati su un video reale.

# Struttura

```
src/image_segmentation/
    board.py            quadrilatero della lavagna, omografia, calibrazione JSON
    person.py           maschera del docente e punto indicato (MediaPipe)
    reconstruction.py   mediana temporale sulla lavagna rettificata
    video.py            lettura del video a passo fisso, mediana dei frame
    main.py             CLI
notebooks/              esplorazione (01 frame singolo, 02 ricostruzione temporale)
tests/                  test end-to-end su video sintetico
scripts/                download dei modelli
data/                   frame, video, calibrazione (fuori da git)
models/                 modelli MediaPipe (fuori da git)
output/                 risultati (fuori da git)
```

# Setup

```
py -m venv .env
.env\Scripts\activate
pip install -e ".[dev]"
python scripts/download_models.py
```

Notebook: `jupyter lab` dalla radice, poi `notebooks/`.
Test: `pytest -s tests` (richiede `data/frame/lezione_prova_cit_censured.png`).

# Uso

```
lavagna data/video/lezione.mp4 --step 1
```

Opzioni principali:
- `--step`: secondi tra due frame analizzati (default 1)
- `--crop x,y,w,h`: area del video, se è una registrazione dello schermo con l'interfaccia del player
- `--quad file.json`: angoli della lavagna fissati a mano
- `--window`: campioni nella mediana temporale (default 7)
- `--min-change`: frazione di pixel cambiati per salvare un nuovo snapshot
- `--refine`: aggiunge DeepLab sul ritaglio alla maschera della posa

Output in `output/<nome-video>/`:
- `board_<hh>h<mm>m<ss>s.png`: snapshot della lavagna pulita
- `board_final.png`, `median.png`, `quad.json`
- `pointing.jsonl`: una riga per frame con il docente in vista

```json
{"t": 693.0, "frame_xy": [679, 415], "board_xy": [612, 251], "board_xy_norm": [0.398, 0.762],
 "arm_angle": 94.0, "occluded_fraction": 0.054, "snapshot": "board_00h11m30s.png"}
```

`arm_angle` è l'angolo tra braccio e busto: circa 0 a riposo, circa 90 con il braccio teso.

# Stato del progetto

Pipeline completa, verificata solo su un frame e su un video sintetico.

Problema noto: sul video sintetico il punto indicato è giusto in 7 frame su 10. In uno la mano esce dal frame; negli altri due PoseLandmarker legge il braccio teso come abbassato, anche su frame quasi identici a uno letto bene. Contromisure da provare sul video reale: modalità VIDEO di MediaPipe (tracking tra frame), indicazione valida solo se persiste per 2 o più campioni.

Prossimi passi:
- provare su un video reale di una lezione intera
- ombre del docente sulla lavagna (non sono nella maschera)
- tarare la soglia degli snapshot e distinguere l'indicare dal gesticolare (angolo e durata della posa)
- esportare per i riassunti: snapshot + intervalli di trascrizione + zone indicate
- interrogazione da timestamp: dato un istante (o una frase della trascrizione), restituire snapshot e zona indicata nella finestra (vedi "Dalla trascrizione alla formula")
- valutare la ritrascrizione con timestamp per parola (Whisper) per stringere la finestra
