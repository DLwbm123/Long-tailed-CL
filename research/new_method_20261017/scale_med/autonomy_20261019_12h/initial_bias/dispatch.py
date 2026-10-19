def main(cfg):
 if cfg['mode'] in ['TRAIN','TRAIN_PREFLIGHT','FEATURE']:
  from train import main
 else:
  from pipeline import main
 main(cfg)
