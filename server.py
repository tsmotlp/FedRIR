import copy
import torch
import time
import numpy as np
from client import FedRIRClient


class FedRIRServer:
    def __init__(self, args):
        self.global_epochs = args.global_epochs
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.eval_interval = args.eval_interval

        self.clients = []

        for i in range(self.num_clients):
            client = FedRIRClient(args, client_id=i)
            self.clients.append(client)

        self.best_test_acc = 0
        self.Budget = []

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_model(self.gfe)

    def add_parameters(self, w, gfe):
        for server_param, client_param in zip(self.gfe.parameters(), gfe.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.gfe = copy.deepcopy(self.uploaded_models[0])
        for param in self.gfe.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, gfe in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, gfe)

    def run(self):
        for i in range(self.global_epochs):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_interval == 0:
                print(f"\n-------------------------------Round number: {i}-------------------------------")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', self.Budget[-1], '-'*25)

        print("Best accuracy: ", self.best_test_acc)
        print("Average time cost per round: ", sum(self.Budget[1:])/len(self.Budget[1:]))

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_models.append(client.model.gfe)
    

    def test_metrics(self):
        num_samples = []
        cls_losses = []
        corrects = []
        
        for c in self.clients:
            test_stats = c.test_metrics()
            corrects.append(test_stats["test_corrects"])
            num_samples.append(test_stats["test_num_samples"])
            cls_losses.append(test_stats["test_cls_loss"])

        return num_samples, corrects, cls_losses

    def train_metrics(self):
        num_samples = []
        recon_losses = []
        cls_losses = []
        id_losses = []
        corrects = []
        for c in self.clients:
            train_stats = c.train_metrics()
            num_samples.append(train_stats["train_num_samples"])
            recon_losses.append(train_stats["train_recon_loss"])
            cls_losses.append(train_stats["train_cls_loss"])
            id_losses.append(train_stats["train_id_loss"])
            corrects.append(train_stats["train_corrects"])
        
        return num_samples, corrects, recon_losses, cls_losses, id_losses

    # evaluate selected clients
    def evaluate(self):
        stats_test = self.test_metrics()
        stats_train = self.train_metrics()

        test_cls_loss = sum(stats_test[2])*1.0 / sum(stats_test[0])
        test_acc = sum(stats_test[1])*1.0 / sum(stats_test[0])

        train_recon_loss = sum(stats_train[2])*1.0 / sum(stats_train[0])
        train_cls_loss = sum(stats_train[3])*1.0 / sum(stats_train[0])
        train_id_loss = sum(stats_train[4])*1.0 / sum(stats_train[0])
        train_acc = sum(stats_train[1])*1.0 / sum(stats_train[0])

        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        accs = [a / n for a, n in zip(stats_test[1], stats_test[0])]

        print(f"[Train] recon Loss: {train_recon_loss:.4f}, cls Loss: {train_cls_loss:.4f}, id Loss: {train_id_loss:.4f}, acc: {train_acc:.4f}")
        print(f"[Test] cls Loss: {test_cls_loss:.4f}, acc: {test_acc:.4f}±{np.std(accs):.4f}")

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients