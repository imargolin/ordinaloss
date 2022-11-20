import mlflow

class Callback:
    def __init__(self):
        pass
    
    def on_train_start(self, engine) -> None:
        raise NotImplementedError

    def on_train_end(self, engine) -> None:
        pass

    def on_eval_start(self, engine) -> None:
        pass

    def on_eval_end(self, engine) -> None:
        pass

    def on_predict_start(self, engine) -> None:
        pass

    def on_predict_end(self, engine) -> None:
        pass

    def on_init(self, engine) -> None:
        pass

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
        print("Model initialized")

class MLFlowCallback(Callback):
    def __init__(self):
        pass

class EarlyStoppingCallback(Callback):
    def on_init(self, engine):
        pass

    def on_train_end(self, engine) -> None:
        return super().on_train_end(engine)

