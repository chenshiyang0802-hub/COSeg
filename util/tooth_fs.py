import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import glob
import time
from itertools import combinations
from itertools import permutations
# from util.data_util import data_prepare_v101 as data_prepare

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def voxelize(coord, voxel_size):
    """ 简单的网格下采样实现 """
    discrete_coord = np.floor(coord / np.array(voxel_size))
    key = discrete_coord.astype(np.int32)
    # 使用 numpy 的 unique 函数找到唯一体素的索引
    _, unique_indices = np.unique(key, axis=0, return_index=True)
    return unique_indices

def tooth_data_prepare(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,  # 
    voxel_max=None,
    transform=None,
    shuffle_index=False,
    sampled_class=None,
):
    # 0. 归一化
    coord = pc_normalize(coord)
    
    # 1. 数据增强 (如果传入了 transform)
    if transform:
        coord, feat = transform(coord, feat)

    # 2. 体素下采样 (Grid Subsampling)
    if voxel_size:
        # 归一化后坐标变小，voxel_size 建议 0.03-0.04
        coord_min = np.min(coord, 0)
        coord -= coord_min
        
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]

    # 3. 点数裁剪 (Random Sampling)
    # 如果点数超过 voxel_max (例如 4096)，随机采样子集
    if voxel_max and label.shape[0] > voxel_max:
        crop_idx = np.random.choice(
            np.arange(label.shape[0]), voxel_max, replace=False
        )
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]

    # 4. 打乱顺序
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    # 5. 转为 Tensor
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)

    return coord, feat, label



class Tooth_base(Dataset):
    def __init__(
        self,
        split="train",
        data_root="trainval", 
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        preload=True,
    ):
        super().__init__()
        (
            self.split,
            self.voxel_size,
            self.transform,
            self.voxel_max,
            self.shuffle_index,
            self.loop,
            self.preload,
        ) = (split, voxel_size, transform, voxel_max, shuffle_index, loop, preload)

        self.data_root = data_root
        
        # -------------------------------------------------------
        # 定义牙齿的类别
        # -------------------------------------------------------
        self.all_classes = [i for i in range(1, 17)] #! 不含 0-gum
        self.class_count = len(self.all_classes)

        # Base: 切牙(1,2,9,10) + 磨牙(6,7,8,14,15,16)
        incisors_and_molars = [1, 2, 6, 7, 8, 9, 10, 14, 15, 16]
        # Novel: 尖牙(3,11) + 前磨牙(4,5,12,13)
        canines_and_premolars = [3, 4, 5, 11, 12, 13]

        # 定义交叉验证的 Fold (将牙齿分为训练集和测试集)
        self.fold_0 = incisors_and_molars
        self.fold_1 = canines_and_premolars

        if cvfold == 0:
            self.test_classes = self.fold_1  # 测试集用 Novel
            self.train_classes = self.fold_0 # 训练集用 Base
        elif cvfold == 1:
            self.test_classes = self.fold_0
            self.train_classes = self.fold_1
        else:
            raise NotImplementedError("Unknown cvfold (%s). [Options: 0,1]" % cvfold)

        self.train_classes = [
            c for c in self.all_classes if c not in self.test_classes
        ]

        # 建立索引：{class_id: [scan_name1, scan_name2, ...]}
        self.class2scans = self.get_class2scans() 

        # 内存缓存
        self.data_cache = {}
        if self.preload:
            self.preload_data()

    def get_class2scans(self):
        root_parent = os.path.dirname(self.data_root)
        class2scans_file = os.path.join(root_parent, "tooth_class2scans.pkl")
        
        if os.path.exists(class2scans_file):
            with open(class2scans_file, "rb") as f:
                class2scans = pickle.load(f)
        else:
            min_pts = 50 # 牙齿数据点可能较少，适当降低阈值
            class2scans = {k: [] for k in self.all_classes}

            # 遍历所有 .npy 文件
            files = glob.glob(os.path.join(self.data_root, "*.npy"))
            print(f"Found {len(files)} files. Building index...")

            for file in files:
                scan_name = os.path.basename(file)[:-4]
                # data shape: N x 7 (XYZ, Nx, Ny, Nz, Label)
                data = np.load(file, mmap_mode='r')
                labels = data[:, 6].astype(int)
                classes = np.unique(labels)
                
                for class_id in classes:
                    if class_id not in self.all_classes:
                        continue # 忽略不在定义列表里的杂类(如0)

                    num_points = np.count_nonzero(labels == class_id)
                    if num_points > min_pts:
                        class2scans[class_id].append(scan_name)

            print("==== class to scans mapping is done ====")
            with open(class2scans_file, "wb") as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        
        return class2scans
    
    def preload_data(self):
        print("Preloading data into memory...")
        files = glob.glob(os.path.join(self.data_root, "*.npy"))
        t0 = time.time()
        for f in files:
            scan_name = os.path.basename(f)[:-4]
            self.data_cache[scan_name] = np.load(f).astype(np.float32)
        print(f"Preloaded {len(self.data_cache)} files in {time.time()-t0:.2f}s")


class Tooth_FS(Tooth_base):
    def __init__(
        self,
        split="train",
        data_root="trainval",
        voxel_size=0.04, #!
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=1000, # 每个 epoch 跑多少个 episode
        n_way=2,      
        k_shot=1,     
        n_queries=1, 
        preload=True 
    ):
        super().__init__(
            split, data_root, voxel_size, voxel_max, transform, shuffle_index, loop, cvfold, preload
        )

        self.n_way, self.k_shot, self.n_queries, self.num_episode = (
            n_way,
            k_shot,
            n_queries,
            num_episode,
        )

        if split == "train":
            self.classes = np.array(self.train_classes)
        elif split == "test":
            self.classes = np.array(self.test_classes)
        else:
            raise NotImplementedError("Unknown mode %s! [Options: train/test]" % split)

        # 映射 Train classes 到连续索引 1..N (用于辅助 Loss)
        self.train_mapping = {c: i + 1 for i, c in enumerate(self.train_classes)}
        print(f"[{split}] Classes: {self.classes}")


    def get_test_episode(self, n_way_classes=None):
        """
        指定类别生成测试数据，返回标准的 5 个变量。
        """
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)

        support_ptclouds, support_masks = [], []
        query_ptclouds, query_labels = [], []

        black_list = []
        
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            # 过滤掉已选的
            valid_scannames = [x for x in all_scannames if x not in black_list]
            
            needs = self.k_shot + self.n_queries
            if len(valid_scannames) < needs:
                selected_scannames = np.random.choice(all_scannames, needs, replace=True)
            else:
                selected_scannames = np.random.choice(valid_scannames, needs, replace=False)
            
            black_list.extend(selected_scannames)
            
            query_scannames = selected_scannames[: self.n_queries]
            support_scannames = selected_scannames[self.n_queries :]

            for scan_name in query_scannames:
                # 【关键点】调用 sample_test_pointcloud，只接收 2 个返回值
                ptcloud, label = self.sample_test_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=False
                )
                query_ptclouds.append(ptcloud)
                query_labels.append(label)

            for scan_name in support_scannames:
                ptcloud, label = self.sample_test_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=True
                )
                support_ptclouds.append(ptcloud)
                support_masks.append(label)

        return (
            support_ptclouds,
            support_masks,
            query_ptclouds,
            query_labels,
            sampled_classes,
        )
    

    def sample_test_pointcloud(self, scan_name, sampled_classes, sampled_class, support):
        """
        仿照 S3DIS_FS.sample_test_pointcloud 实现。
        只返回 feat 和 test_label (2个值)。
        """
        # 1. 读取数据 (复用原有的缓存逻辑)
        if self.preload and scan_name in self.data_cache:
            data = self.data_cache[scan_name]
        else:
            data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord = data[:, 0:3]
        feat = data[:, 3:6]
        label = data[:, 6]

        # 2. 数据预处理
        coord, feat, label = tooth_data_prepare(
            coord, feat, label,
            self.split, self.voxel_size, self.voxel_max,
            self.transform, self.shuffle_index, sampled_class
        )

        # 3. 拼接特征
        feat = torch.cat((coord, feat), dim=1)

        # 4. 生成 Test Label (仅 Few-shot 任务需要)
        if support:
            # Support Set: 目标类为1，其余为0
            label = (label == sampled_class).int()
        else:
            # Query Set: 将 sampled_classes 映射为 1..N
            class_dict = {c: i + 1 for i, c in enumerate(sampled_classes)}
            new_label = torch.zeros_like(label)
            
            for i, lb in enumerate(label):
                if lb.item() in class_dict.keys():
                    new_label[i] = class_dict[lb.item()]
                else:
                    new_label[i] = 0
            label = new_label

        return feat, label
    

    def __getitem__(self, idx):

        sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)

        support_ptclouds, support_base_masks, support_test_masks = [], [], []
        query_ptclouds, query_base_labels, query_test_labels = [], [], []

        black_list = []
        
        for sampled_class in sampled_classes:
            # 获取该类的所有文件名
            all_scannames = self.class2scans[sampled_class].copy()
            valid_scannames = [x for x in all_scannames if x not in black_list]
            
            # 容错：如果样本不足，允许重复
            needs = self.k_shot + self.n_queries
            if len(valid_scannames) < needs:
                selected_scannames = np.random.choice(all_scannames, needs, replace=True)
            else:
                selected_scannames = np.random.choice(valid_scannames, needs, replace=False)
            
            black_list.extend(selected_scannames)
            
            # 划分 Support / Query
            query_scannames = selected_scannames[: self.n_queries]
            support_scannames = selected_scannames[self.n_queries :]

            # 构建 Query
            for scan_name in query_scannames:
                ptcloud, base_label, test_label = self.sample_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=False
                )
                query_ptclouds.append(ptcloud)
                query_base_labels.append(base_label)
                query_test_labels.append(test_label)

            # 构建 Support
            for scan_name in support_scannames:
                ptcloud, base_label, test_label = self.sample_pointcloud(
                    scan_name, sampled_classes, sampled_class, support=True
                )
                support_ptclouds.append(ptcloud)
                support_base_masks.append(base_label)
                support_test_masks.append(test_label)

        return (
            support_ptclouds,
            support_base_masks,
            support_test_masks,
            query_ptclouds,
            query_base_labels,
            query_test_labels,
            sampled_classes,
        )

    def sample_pointcloud(self, scan_name, sampled_classes, sampled_class, support):
        # 1. 读取数据 (优先从内存读取)
        if self.preload and scan_name in self.data_cache:
            data = self.data_cache[scan_name] # shape: N x 7
        else:
            data = np.load(os.path.join(self.data_root, scan_name + ".npy"))

        coord = data[:, 0:3] # X, Y, Z
        feat = data[:, 3:6]  # Nx, Ny, Nz
        label = data[:, 6]   # Label (0-16)

        # 2. 调用针对牙齿优化的 prepare 函数
        coord, feat, label = tooth_data_prepare(
            coord, feat, label,
            self.split, self.voxel_size, self.voxel_max,
            self.transform, self.shuffle_index, sampled_class
        )

        # 3. 拼接特征 (XYZ + Normals) -> 6 Channels
        feat = torch.cat((coord, feat), dim=1)

        # 4. 生成 Labels
        # Base Label (用于辅助任务: 识别所有训练集见过的牙齿)
        # Gum(0) 和其他非 Base 的牙齿都会变成 0，被模型视为“背景”安全处理
        train_label = torch.zeros(label.shape, dtype=torch.long)

        for real_id, mapped_id in self.train_mapping.items():
            # 将原标签中等于 real_id 的位置，设为 mapped_id
            train_label[label == real_id] = mapped_id

        # Test Label (Few-shot 任务核心)
        if support:
            # Support: 当前类为1，其他为0 (包括 Gum)
            test_label = (label == sampled_class).int()
        else:
            test_label = torch.zeros(label.shape, dtype=torch.long)
            test_mapping = {c: i + 1 for i, c in enumerate(sampled_classes)}

            if self.split == "train":
                for c in self.test_classes:
                    # 如果某个点属于测试集类别(干扰项)，设为 0
                    test_label[label == c] = 0
            for real_id, mapped_id in test_mapping.items():
                test_label[label == real_id] = mapped_id

        return feat, train_label, test_label

    def __len__(self):
        return self.num_episode


class Tooth_FS_TEST(Dataset):
    def __init__(
        self,
        split="test",  # 注意这里通常是 test 或 val
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=1000,
        n_way=2,
        k_shot=1,
        n_queries=1,
        num_episode_per_comb=100, # 每个类别组合测试多少个 episode
        preload=True
    ):
        super().__init__()

        self.dataset = Tooth_FS(
            split="test", 
            data_root=data_root,
            voxel_size=voxel_size,
            voxel_max=voxel_max,
            transform=transform,
            shuffle_index=shuffle_index,
            loop=loop,
            cvfold=cvfold,
            num_episode=num_episode,
            n_way=n_way,
            k_shot=k_shot,
            n_queries=n_queries,
        )
        
        self.classes = self.dataset.classes
        self.n_way = n_way
        self.num_episode_per_comb = num_episode_per_comb

        self.cvfold = cvfold
        self.k_shot = k_shot
        self.voxel_size = voxel_size
        self.data_root = data_root

        # 定义保存测试数据的文件夹路径
        # 格式参考 S3DIS: S_{fold}_N_{way}_K_{shot}_...
        self.test_data_path = os.path.join(
            os.path.dirname(data_root),
            "Tooth_S_%d_N_%d_K_%d_test_episodes_%d_vs_%.2f"
            % (
                cvfold,
                n_way,
                k_shot,
                num_episode_per_comb,
                voxel_size,
            ),
        )
        self.prepare_test_data()


    def prepare_test_data(self):

        if os.path.exists(self.test_data_path):
            self.file_names = glob.glob(os.path.join(self.test_data_path, "*.pt"))
            self.num_episode = len(self.file_names)
            print(f"Loading existing test data from {self.test_data_path}, found {self.num_episode} episodes.")
        else:
            print(f"Test dataset ({self.test_data_path}) does not exist...\n Constructing...")
            os.mkdir(self.test_data_path)

            class_comb = list(combinations(self.classes, self.n_way))
            self.num_episode = len(class_comb) * self.num_episode_per_comb

            episode_ind = 0
            self.file_names = []

            print(f"Target: {len(class_comb)} combinations, {self.num_episode_per_comb} episodes each.")

            # 直接遍历组合，确定性生成数据
            for sampled_classes in class_comb:
                for _ in range(self.num_episode_per_comb):
                    
                    # 直接调用 get_test_episode，传入指定类别
                    # 这一步必定成功返回包含该组合的 5 个变量的数据
                    data = self.dataset.get_test_episode(n_way_classes=sampled_classes)
                    
                    out_filename = os.path.join(self.test_data_path, f"{episode_ind}.pt")
                    write_episode(out_filename, data)
                    self.file_names.append(out_filename)
                    episode_ind += 1
            
            print(f"Generation Complete. Total episodes saved: {self.num_episode}")

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index):
        file_name = self.file_names[index]
        return read_episode(file_name)
    
def write_episode(out_filename, data):
    """ 辅助函数：将生成的 Episode 保存到硬盘 """
    support_feat, support_label, query_feat, query_label, sampled_classes = data
    torch.save(
        {
            "support_feat": support_feat,
            "support_label": support_label,
            "query_feat": query_feat,
            "query_label": query_label,
            "sampled_classes": sampled_classes,
        },
        out_filename,
    )
    print("\t {0} saved! | classes: {1}".format(out_filename, sampled_classes))

def read_episode(file_name):
    """ 和 s3dis的保持一致 """
    data_file = torch.load(file_name)
    return (
        data_file["support_feat"],
        data_file["support_label"],
        data_file["query_feat"],
        data_file["query_label"],
        data_file["sampled_classes"],
    )


class Tooth_FSForVIS(Tooth_FS):
    """
    用于可视化 Few-shot 结果的 Tooth 数据集类
    针对特定的 target_class
    遍历所有可能的 (Support, Query) 组合
    """

    def __init__(
        self,
        split="test",  # 通常可视化是在测试集上做
        data_root="trainval",
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        cvfold=0,
        num_episode=1000, # 这里其实用不到，因为长度由排列组合决定
        n_way=1,      # 可视化通常是一对一 (1-way)
        k_shot=1,     # 通常是一对一 (1-shot)
        n_queries=1,  # 通常是一对一 (1-query)
        preload=True,
        target_class=None, # 指定要可视化的牙齿 ID (例如 1 代表中切牙)
    ):
        super().__init__(
            split,
            data_root,
            voxel_size,
            voxel_max,
            transform,
            shuffle_index,
            loop,
            cvfold,
            num_episode,
            n_way,
            k_shot,
            n_queries,
            preload,
        )

        self.target_class = target_class
        
        # 检查 target_class 是否在当前 split 的类别中
        if self.target_class not in self.class2scans:
             raise ValueError(f"Target class {self.target_class} not found in dataset index.")

        # 获取该目标类别下所有可用的扫描文件名
        available_scans = self.class2scans[self.target_class]

        # 生成所有可能的组合 (Support, Query)
        # 长度为 k_shot + n_queries (通常是 1+1=2)
        combo_length = self.k_shot + self.n_queries
        if len(available_scans) < combo_length:
             print(f"Warning: Not enough scans for class {self.target_class} to make unique pairs.")
             self.combos = []
        else:
            # 排列组合：从可用扫描中取出 combo_length 个不重复的样本
            self.combos = list(permutations(available_scans, combo_length))
        
        print(f"[VIS Dataset] Target Class: {self.target_class} | Total Combinations: {len(self.combos)}")

    def __getitem__(self, idx):
        """
        返回指定索引的 Support 和 Query 数据对
        """
        support_ptclouds, support_masks = [], []
        query_ptclouds, query_labels = [], []

        # 获取当前组合的文件名列表
        selected_scannames = self.combos[idx]
        
        # 切分 Query 和 Support
        # 注意：这里通常 Query 在前还是 Support 在前取决于你的习惯
        # S3DIS代码逻辑是：先切出 query，剩下的给 support
        query_scannames = selected_scannames[: self.n_queries]
        support_scannames = selected_scannames[self.n_queries :]

        # 1. 构建 Query Set
        for scan_name in query_scannames:
            ptcloud, label = self.sample_test_pointcloud(
                scan_name, [self.target_class], self.target_class, support=False
            )
            query_ptclouds.append(ptcloud)
            query_labels.append(label)

        # 2. 构建 Support Set
        for scan_name in support_scannames:
            ptcloud, label = self.sample_test_pointcloud(
                scan_name, [self.target_class], self.target_class, support=True
            )
            support_ptclouds.append(ptcloud)
            support_masks.append(label)

        return (
            support_ptclouds,
            support_masks,
            query_ptclouds,
            query_labels,
            np.array([self.target_class]), # 返回当前可视化的类别 ID
            selected_scannames,            # 返回文件名，方便保存图片命名
        )

    def __len__(self):
        return len(self.combos)