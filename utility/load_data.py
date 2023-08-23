import os
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from numpy import ndarray


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        with open(path + '/train.txt') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(path + '/test.txt') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.S = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(path + '/train.txt') as f_train:
            with open(path + '/test.txt') as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        self.U = self.R.dot(self.R.transpose())
        self.I = self.R.transpose().dot(self.R)

        self.n_social = 0
        if os.path.exists(path + '/social_trust.txt'):
            with open(path + '/social_trust.txt') as f_social:
                for l in f_social.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    users = [int(i) for i in l.split(' ')]
                    uid, friends = users[0], users[1:]
                    for i in friends:
                        self.S[uid, i] = 1.
                        self.n_social = self.n_social + 1

    def get_norm_adj_mat(self):
        def normalized_sym(adj):
            rowsum: ndarray = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocsr()

        try:
            t1 = time()
            interaction_adj_mat_sym = sp.load_npz(self.path + '/s_interaction_adj_mat.npz')
            print('already load interaction adj matrix', interaction_adj_mat_sym.shape, time() - t1)
        except Exception:
            interaction_adj_mat = self.create_interaction_adj_mat()
            interaction_adj_mat_sym = normalized_sym(interaction_adj_mat)
            print('generate symmetrically normalized interaction adjacency matrix.')
            sp.save_npz(self.path + '/s_interaction_adj_mat.npz', interaction_adj_mat_sym)
        try:
            t2 = time()
            social_adj_mat_sym = sp.load_npz(self.path + '/s_social_adj_mat.npz')
            print('already load social adj matrix', social_adj_mat_sym.shape, time() - t2)
        except Exception:
            social_adj_mat = self.create_social_adj_mat()
            social_adj_mat_sym = normalized_sym(social_adj_mat)
            print('generate symmetrically normalized social adjacency matrix.')
            sp.save_npz(self.path + '/s_social_adj_mat.npz', social_adj_mat_sym)
        try:
            t3 = time()
            similar_users_adj_mat_sym = sp.load_npz(self.path + '/s_similar_users_adj_mat.npz')
            print('already load similar users adj matrix', similar_users_adj_mat_sym.shape, time() - t3)
        except Exception:
            similar_users_adj_mat = self.create_similar_adj_mat()
            similar_users_adj_mat_sym = normalized_sym(similar_users_adj_mat)
            print('generate symmetrically normalized similar users adjacency matrix.')
            sp.save_npz(self.path + '/s_similar_users_adj_mat.npz', similar_users_adj_mat_sym)

        return interaction_adj_mat_sym, social_adj_mat_sym, similar_users_adj_mat_sym

    def create_interaction_adj_mat(self):
        # 1. Create Graph Users-Items  Interaction Adjacency Matrix.
        t1 = time()
        interaction_adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                                            dtype=np.float32)
        interaction_adj_mat = interaction_adj_mat.tolil()
        R = self.R.tolil()
        for i in range(5):
            interaction_adj_mat[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5), self.n_users:] = \
                R[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)]
            interaction_adj_mat[self.n_users:, int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)] = \
                R[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)].T
        print('already create interaction adjacency matrix', interaction_adj_mat.shape, time() - t1)
        return interaction_adj_mat.tocsr()

    def create_social_adj_mat(self):
        # 2. Create Graph Users-Users Social Adjacency Matrix.
        t2 = time()
        social_adj_mat = self.S
        print('already create social adjacency matrix', social_adj_mat.shape, 'social_interactons:', self.n_social,
              time() - t2)
        return social_adj_mat.tocsr()

    def create_similar_adj_mat(self):
        t3 = time()
        similar_users_adj_mat = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)

        def cluster(x, d, t):
            v = x / (t + d - x)
            if v <= 0.09:
                return 0
            elif 0.09 < v <= 0.39:
                return 1
            elif 0.39 < v <= 0.49:
                return 10
            elif 0.49 < v <= 0.79:
                return 100
            else:
                return 200

        X = self.U.toarray()
        vfunc = np.vectorize(cluster)

        diag: ndarray = X.diagonal()
        for i in range(X.shape[0]):
            tmp = vfunc(X[i], diag, diag[i])
            similar_users_adj_mat[i] = tmp
        print('already create similar users adjacency matrix', similar_users_adj_mat.shape, time() - t3)
        similar_users_adj_mat = sp.csr_matrix(similar_users_adj_mat)

        return similar_users_adj_mat.tocsr()

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))
