# CLAUDE.md

Toolkit in Python per la segmentazione di immagini e video: separare soggetti dallo sfondo, ricostruire ciò che coprono, seguire gesti e posizioni nel tempo. È un progetto **generale**: ogni applicazione concreta è un caso d'uso costruito sopra moduli riusabili.

## Stato e compito

Il primo caso d'uso è già implementato: **lavagna senza docente**. Dai video delle lezioni toglie la docente, salva la lavagna pulita a ogni cambiamento e registra il punto indicato con il timestamp. È descritto nel `README.md`, in `notebooks/` e in `tests/test_synthetic.py`. Il codice in `src/image_segmentation/` e il comando `lavagna` sono scritti per quel caso.

Il compito è rendere il progetto **modulabile** e impacchettare il caso della lavagna come primo caso d'uso. La struttura la decidi tu: pacchetti, interfacce, come si compongono i passi, come si cambiano i modelli, nomi e CLI. Il codice esistente è materiale da riorganizzare, non uno scheletro da rispettare. Per lo stesso motivo il README, oggi intitolato alla lavagna, va riscritto per il progetto generale.

Prima di spostare o rinominare il codice esistente, proponi la struttura a Lorenzo e aspetta il suo ok. Il formato di output del caso lavagna (snapshot `board_<hh>h<mm>m<ss>s.png` e `pointing.jsonl`) lo usa il vault: se cambia, va detto.

**Lezioni del primo caso d'uso**, utili oltre la lavagna:
- I modelli con input piccolo (DeepLab V3 a 257×257) perdono i soggetti piccoli nel frame intero: si ritaglia prima, oppure si usa un modello a due passi come PoseLandmarker.
- Uno sfondo statico con un soggetto in movimento si ricostruisce con la mediana temporale dei pixel non coperti, dopo averlo portato in un piano comune con un'omografia.
- Il colore distingue bene uno sfondo uniforme, ma non soggetti chiari su tratti chiari: serve come rifinitura, non come criterio principale.
- I landmark della posa danno più di una maschera: per esempio il punto indicato e l'angolo del braccio.

Cosa è specifico e cosa è generico nel codice attuale:
- `board.py`: quadrilatero di una lavagna;
- `person.py`: persona con MediaPipe;
- `reconstruction.py`: sfondo statico su un piano rettificato;
- `video.py`: lettura dei frame.

È un punto di partenza per decidere i confini, non una decisione già presa.

## Dati

- `data/`, `models/` e `output/`, insieme a immagini e video, sono fuori da git (`.gitignore`). I frame delle lezioni mostrano una persona reale: non vanno mai committati, pubblicati o allegati a issue e PR.
- I notebook si committano **senza output**.
- Test ed esempi pubblici usano dati sintetici.
- `tests/test_synthetic.py` richiede `data/frame/lezione_prova_cit_censured.png` e senza quel file viene saltato.
- Il vault universitario `D:\Uniud\ms-aics` fornisce le fonti ed è **fuori perimetro**: si legge solo se Lorenzo lo chiede, non ci si scrive mai.

## Ambiente

- Windows, Git Bash. `python` non è nel PATH: usa `.env\Scripts\python` oppure il launcher `py`.
- Versioni: Python 3.12, OpenCV 5.0, NumPy 2.x, MediaPipe Tasks 1.1.1.
- Si lavora su CPU. La GPU (GTX 1050 Ti, 4 GB) non è usata.

```
py -m venv .env
.env\Scripts\activate
pip install -e ".[dev]"
python scripts/download_models.py   # modelli MediaPipe in models/
pytest -s tests
jupyter lab                          # dalla radice, poi notebooks/
```

**MediaPipe:** le maschere si leggono sempre come `.numpy_view().squeeze().copy()`. La vista punta alla memoria del risultato: se il risultato è temporaneo, l'accesso successivo provoca un crash nativo (kernel Jupyter morto, access violation), non un'eccezione.

## Convenzioni

- **Lingua:** italiano per README, docstring e commenti; i termini tecnici restano in inglese. Commit in inglese, conventional commits con scope (`feat(notebook): ...`), come in `D:\Programmazione\Projects\image-stitching`, il progetto fratello da cui vengono layout e stile.
- **Git:** niente commit senza il «sì» di Lorenzo. Proponi file e messaggio e aspetta. Nel commit va la riga `Co-Authored-By` di Claude. File aggiunti per nome, mai `git add -A`. Mai `push --force` né `--no-verify`. Nessun remote né `gh repo create` senza conferma.
- Il contesto della sessione che ha creato la repo (esperimenti, numeri, motivazioni) è nella memoria di Claude per questo progetto.
