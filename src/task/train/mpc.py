import logging
import numpy as np
import random
import wandb

import torch
from torch.utils.data import DataLoader

from data.base import newsample
from task.train.user import UserTrainTask
from mpc import TrustThirdParty, MPCServer, MPCUser
from data import get_dataset, get_collate_fn

class MPCTrainTask(UserTrainTask):
   def __init__(self, args, config, device):
      super().__init__(args, config, device)

      self.user_index = {uid: uindex for uid, uindex in zip(self.user_indices, range(len(self.user_indices)))}
      self.init_mpc()

   def init_mpc(self):
      logging.debug("----------- init mpc -----------")
      logging.debug("start init trusted third party")
      self.ttp = TrustThirdParty()
      logging.debug("end init trusted third party")

      logging.debug("start init mpc server")
      self.server = MPCServer(self.args.user_num, self.args.mpc_t)
      logging.debug("end init mpc server")

      logging.debug("start generate sign keys for users")
      self.ttp.gen_keys(list(self.user_index.values()))
      logging.debug("end generate sign keys for users")

      logging.debug("start init MPCUser class")
      MPCUser.initialize(self.args.user_num, self.args.mpc_t, self.server, self.ttp)
      logging.debug("end init MPCUser class")
      MPCUser.get_dh_keys()
   
   def collect_users_nids(self, train_sam, users, user_indices, nid2index):
      user_sample = 0

      raw_vecs = {}

      # get news id_vecs of all user and send to MPC server
      for user in users:
         uindex = self.user_index[user]
         sids = user_indices[user]
         user_sample += len(sids)
         user_news = set([0])

         news_id_vec = np.zeros(len(nid2index), dtype='int32')
         for idx in sids:
            _, pos, neg, his, _ = train_sam[idx]
            user_news = user_news | set([nid2index[i] for i in [pos] + neg + his])
         user_news = list(user_news)
         news_id_vec[user_news] = np.random.randint(low=1, high=10, size=len(user_news))
         raw_vecs[uindex] = {"news_id_vec": news_id_vec}

         user_instance = MPCUser(uindex, raw_vecs[uindex])
         user_instance.send_pub_keys()

      user_nids = []
      summed_news_id_vec = self.server.unmask_vecs["news_id_vec"].round().astype("int32")
      self.server.clear()

      for i in range(len(summed_news_id_vec)):
         if summed_news_id_vec[i] >= 1:
            user_nids.append(i)

      union_nid_index = {nid: nindex for nid, nindex in zip(user_nids, range(len(user_nids)))}

      return user_nids, union_nid_index, user_sample

   def get_sampled_dataset(self, train_sam, users, user_indices, nid2index):
      sampled_data = []
      sampled_user_indices = {}

      index = 0
      for user in users:
         sids = user_indices[user]
         sampled_user_indices[user] = []
         for idx in sids:
            imp_id, pos, neg, his, uid = train_sam[idx]
            neg = newsample(neg, self.args.npratio)
            sampled_data.append((imp_id, pos, neg, his, uid))
            sampled_user_indices[user].append(index)
            index += 1

      return sampled_data, sampled_user_indices

   def process_news_grad(self, candidate_info, his_info, union_nid_index, user_mask_matrix):
      candidate_news, candidate_grad = candidate_info
      his, his_grad = his_info

      batch_size = candidate_news.shape[0]
      news_grad = torch.zeros(batch_size, len(union_nid_index), 400).to(self.device)
      
      for b in range(batch_size):
         for i, nid in enumerate(candidate_news[b]):
            nindex = union_nid_index[nid]
            news_grad[b][nindex] += candidate_grad[b][i]
         
         for i, nid in enumerate(his[b]):
            nindex = union_nid_index[nid]
            news_grad[b][nindex] += his_grad[b][i]

      news_grad = torch.einsum(
         "ub,b...->u...", user_mask_matrix, news_grad
      ).cpu()

      return news_grad

   def train_on_step(self, step):
      users = random.sample(self.user_indices.keys(), self.args.user_num)
      sampled_data, sampled_user_indices = self.get_sampled_dataset(self.train_sam, users, self.user_indices, self.nid2index)
      nids, union_nid_index, user_sample = self.collect_users_nids(sampled_data, users, sampled_user_indices, self.nid2index)
      self.agg.gen_news_vecs(nids)

      train_ds = get_dataset(self.config["train_dataset_name"])(
            self.args,
            sampled_data,
            users,
            self.nid2index,
            self.agg,
            self.news_index
      )
      train_collate_fn = get_collate_fn(self.config["train_collate_fn_name"])
      train_dl = DataLoader(
            train_ds,
            batch_size=len(train_ds),
            shuffle=False,
            num_workers=0,
            collate_fn=train_collate_fn,
      )
      self.module._module.user_encoder.load_state_dict(self.agg.user_encoder.state_dict())

      self.module.train()
      loss = 0

      # get update of benigh users
      for cnt, data in enumerate(train_dl):
            sample_num = data["batch_label"].shape[0]
            for key in ["batch_candidate_news_vecs", "batch_his_vecs", "batch_label", "user_mask_matrix"]:
               if torch.is_tensor(data[key]):
                  data[key] = data[key].to(self.device)

            data["batch_candidate_news_vecs"].requires_grad = True
            data["batch_his_vecs"].requires_grad = True

            bz_loss, y_hat = self.module(data)

            loss += bz_loss.detach().cpu().numpy()
            bz_loss.backward()

            candaidate_grad = data["batch_candidate_news_vecs"].grad.detach() * sample_num 
            candidate_news = data["batch_candidate_news"].numpy()

            his_grad = data["batch_his_vecs"].grad.detach() * sample_num 
            his = data["batch_his"].numpy()

            news_grad = self.process_news_grad(
               [candidate_news, candaidate_grad], [his, his_grad], union_nid_index, data["user_mask_matrix"]
            )

            user_grad, user_sample_num = self.process_user_grad(
               self.module._module.user_encoder, data["user_mask_matrix"]
            )
            self.agg.collect(user_grad, user_sample_num, news_grad, union_nid_index)
            self.module.zero_grad(set_to_none=True)

      loss = loss / (cnt + 1)
      self.agg.update(user_sample)

      if "debug" not in self.args.run_name:
            wandb.log({"train loss": loss}, step=step + 1)
