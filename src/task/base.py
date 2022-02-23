from pathlib import Path
import logging

class BaseTask:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device

        self.data_path = Path(args.data_path) / args.data
        self.out_path = Path(args.out_path) / f"{args.run_name}-{args.data}"
        self.out_model_path = self.out_path / "model"

        self.out_model_path.mkdir(exist_ok=True, parents=True)

        self.load_data()
        logging.info("[-] finish loading data.")

        self.load_model()
        logging.info("[-] finish loading model.")

    def load_data(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def start(self):
        return NotImplementedError