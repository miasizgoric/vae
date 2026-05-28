# Autoenkoder — Biomedicinski sažeci (HR)

> Nenadzirani sustav za semantičko rangiranje hrvatskih akademskih biomedicinskih sažetaka s Hrčka i Dabra.

---

## Struktura projekta

```
autoenkoder-hr-bio/
├── Dockerfile                       # PyTorch 2.11 + CUDA 12.8 + HR NLP stack
├── docker-compose.yml               # dev (lokalno) i gpu (server) servisi
├── requirements.txt
├── verify_environment.py            # provjera ispravnosti okruženja
│
├── configs/
│   └── hrcak_bio/
│       ├── ntm_baseline.yaml        # NTM bez disentanglementa
│       └── ntm_hfvae.yaml           # NTM + HFVAE (β=10, γ=5)
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py         # classla lematizacija + BoW
│   │   └── dataset.py               # PyTorch Dataset/DataLoader
│   ├── models/
│   │   ├── vae.py                   # NVDM / NTM / GSM arhitekture
│   │   └── trainer.py               # petlja treniranja + TensorBoard
│   ├── evaluation/
│   │   ├── npmi.py                  # NPMI evaluacija tema
│   │   └── ranking.py               # semantičko rangiranje sažetaka
│   └── utils/
│       ├── config.py                # YAML učitavanje
│       └── seed.py                  # reproducibilnost
│
├── scripts/
│   ├── train.py                     # pokretanje treniranja
│   └── evaluate.py                  # NPMI + rangiranje
│
├── notebooks/                       # Jupyter eksperimenti
├── resources/datasets/              # podaci (nisu u Gitu)
└── results/                         # modeli i rezultati (nisu u Gitu)
```

## Reference

- Ivanković, D. (2020). *Disentangled Representation Learning with Applications in NLP*. FER Zagreb.
- Miao et al. (2016). NVDM. ICML.
- Miao et al. (2017). GSM. arXiv.
- Srivastava & Sutton (2017). ProdLDA/AVITM. arXiv.
- Lau et al. (2014). NPMI topic coherence. EACL.
