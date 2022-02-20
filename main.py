from src.factory import get_trainer

if __name__ == "__main__":
    base_path = '../imagewoof2/'
    csv_name = 'noisy_imagewoof.csv'
    LR = 0.001

    preprocess_conf = {"base_path": base_path, "csv_name": csv_name, "label_name": "noisy_labels_50"}

    train_conf = {"data_path": base_path, "batch_size": 32, "num_workers": 4}
    val_conf = {"data_path": base_path, "batch_size": 32, "num_workers": 4}
    logger_conf = {"path": "./logs/second_try/"} 
    trainer_conf = {"max_epochs": 20, "log_every_n_steps": 3}

    trainer, model, preprocess, train_loader, test_loader = get_trainer('resnet50', LR, preprocess_conf, train_conf, val_conf, logger_conf, trainer_conf)

    trainer.fit(model, train_loader, test_loader)
