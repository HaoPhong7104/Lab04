from trainer import SparkConfig, Trainer
from models import SVM
from transforms import Transforms, Normalize

transforms = Transforms([
    Normalize(mean=(0.1307,), std=(0.3081,))
])

if __name__ == "__main__":
    spark_config = SparkConfig()

    svm = SVM(loss="squared_hinge", penalty="l2")
    trainer = Trainer(svm, "train", spark_config, transforms)
    trainer.train()
    # trainer.predict()
