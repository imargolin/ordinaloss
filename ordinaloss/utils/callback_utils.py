import mlflow

class Callback:
    def __init__(self):
        pass
    
    def on_train_start(self, engine) -> None:
        raise NotImplementedError

    def on_train_end(self, engine) -> None:
        raise NotImplementedError

    def on_eval_start(self, engine) -> None:
        raise NotImplementedError

    def on_eval_end(self, engine) -> None:
        raise NotImplementedError

    def on_predict_start(self, engine) -> None:
        raise NotImplementedError

    def on_predict_end(self, engine) -> None:
        raise NotImplementedError

    def on_init(self, engine) -> None:
        raise NotImplementedError

class PrintingCallback(Callback):
    def on_train_start(self, engine) -> None:
        print("Starting training")

    def on_train_end(self, engine) -> None:
        print("Ending training")

    def on_eval_start(self, engine) -> None:
        print("Starting evaluation")

    def on_eval_end(self, engine) -> None:
        print("Ending evaluation")

    def on_predict_start(self, engine) -> None:
        print("Starting prediction")

    def on_predict_end(self, engine) -> None:
        print("Ending prediction")
       
    def on_init(self, engine) -> None:
        raise NotImplementedError

class MLFlowCallback(Callback):
    def __init__(self):
        pass


