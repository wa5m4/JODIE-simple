# synthetic.md — 对应源码 [synthetic.py](../../jodie/data/synthetic.py)

> **定位**:数据层的基石。定义项目的"原子单位" `Interaction`,并提供**合成数据生成器**(用于快速调试)和**动态图状态容器**(给带图聚合的模型用)。

---

## 🔑 Interaction——项目的原子(源码 12-22 行)

```python
@dataclass
class Interaction:
    """一条交互记录"""

    timestamp: float # 交互发生的时间戳
    user_id: int # 用户ID
    item_id: int # 商品ID
    features: torch.Tensor # 交互特征向量
    neg_samples_by_epoch: Dict[int, List[int]] = field(default_factory=dict)
```

**逐字段白话**:

- `timestamp / user_id / item_id` —— 你在第 2 课已经认识:"谁、和谁、什么时候"。
- `features` —— 这次交互本身的特征向量(第 2 课 C2 已纠正过:是**交互级**特征,不是用户/物品的静态属性)。
- `neg_samples_by_epoch` —— **本项目独有的关键字段**,默认是空字典,不传也没事。

📄 **论文关键点:为什么负样本要预生成?**

看 20-22 行注释:"在数据加载时一次性生成,训练时直接读取,避免 Pipeline 各分区 RNG 重置导致偏差"。翻译成白话:

> 训练一条交互需要给它配一个"陪练"负样本(第 4 课 BPR 讲过)。如果训练时才现场随机抽陪练,那么 Serial 模式和 Pipeline 模式**抽到的陪练不一样**(因为进程数不同、随机数状态不同)→ 两种模式实际上在训练**不同的数据** → 分数自然对不上。解法:在**数据加载阶段**就把每个 epoch 的陪练全部抽好、写死在数据里,任何模式读到的数据**完全一样**。

这就是 ATTEMPTS_LOG 里"test_precomputed_negs.py"对应的修复思想:**训练中的一切随机性,要么冻结成数据,要么各模式一致**。ATTEMPTS_LOG 也记录了:这个修复"必要但不充分"——它消除了负样本差异,但 Pipeline 评分偏差的主因是 optimizer 动量断裂(执行层再细讲)。

---

## 🔑 generate_synthetic_data——合成数据生成器(源码 25-85 行)

**作用**:在没有真实数据时,生成"带用户偏好"的假交互流,用来快速跑通流程。

### 第一部分:RNG 状态的保存/恢复(源码 36-41、80-82 行)

```python
# 保存全局随机数生成器状态，在生成之后恢复
np_rng_state = np.random.get_state()
torch_rng_state = torch.get_rng_state()

np.random.seed(seed) # 设置随机数种子
torch.manual_seed(seed) # 设置PyTorch随机数种子
```

```python
# 恢复全局随机数生成器状态，避免对调用者产生副作用
np.random.set_state(np_rng_state)
torch.set_rng_state(torch_rng_state)
```

**为什么要这样?** 这个函数内部的随机抽取会"消耗"全局随机数。如果不恢复,调用者(比如 NAS 控制器)接下来抽随机数时,结果就会被这个函数**悄悄改变**——调试时表现为"同样代码两次运行结果不同"。保存→使用→恢复,是科研代码的标配习惯:**函数不能污染别人的随机数**。

### 第二部分:用户偏好生成(源码 43-52 行)

```python
num_types = 10 # 物品类型数量
item_type = np.random.randint(0, num_types, num_items) # 每个物品随机分配一个类型

user_type_prefs: Dict[int, Set[int]] = {}
# 为每个用户随机分配一些物品类型作为偏好
for uid in range(num_users):
    n_types = np.random.randint(2, 4)
    user_type_prefs[uid] = set(np.random.choice(num_types, n_types, replace=False))
```

每个物品被随机分到 10 种"类型"之一;每个用户随机喜欢 2~3 种类型。**这模拟了现实:用户有口味偏好,物品有品类**。

### 第三部分:交互生成(源码 54-78 行)

```python
if np.random.random() < 0.8: # 80%偏好驱动
    allowed_types = user_type_prefs[uid]
    candidates = [iid for iid in range(num_items) if item_type[iid] in allowed_types]
    iid = int(np.random.choice(candidates)) if candidates else np.random.randint(0, num_items)
else: # 20%随机选择
    iid = np.random.randint(0, num_items)
```

**80/20 法则**:80% 的交互从用户偏好的类型里选物品(信号),20% 完全随机(噪声)。这样生成的数据里"用户偏好"是可以被模型学到的规律——如果模型训练后 `recall_by_type`(按类型召回)很高,说明它真的学到了偏好。

### 返回三个东西(源码 84 行)

`interactions`(交互流)、`user_type_prefs`(用户偏好,供按类型评估用)、`item_type`(物品类型标签)。

---

## 📖 init_dynamic_graph_state / clone / snapshot / restore(源码 88-133 行)

四个函数围绕同一个数据结构——**动态图状态**:

```python
{
    "num_users": ..., "num_items": ..., "max_neighbors": ...,
    "adj": {},              # 邻接表:node_id -> [邻居列表]
    "edge_last_time": {},   # 每条边最后出现的时间
    "edge_weight": {},      # 每条边的权重
}
```

- `init_...` 建一个空模板;`clone_...` 复制配置、清空数据(保证各次训练互不干扰)
- `snapshot_...` / `restore_...` 做**深拷贝**,供流水线跨进程传递状态用

**现在只需要知道**:这是给"带图聚合"的模型(HybridJODIE)用的公共数据结构;纯 RNN 模型(JODIERNN)不碰它。等模型层讲图聚合时,这里会再回头看。

---

## ❓ 检查题

**D1.** 如果删掉 `generate_synthetic_data` 里"保存/恢复 RNG 状态"的代码,会发生什么现象?为什么调试时特别让人头疼?

答:会出现全局随机数被消耗，后面使用随机数结果不一样



**D2.** 用你自己的话说:`neg_samples_by_epoch` 把负样本"冻"在数据里的核心目的是什么?(提示:对比"训练时现场抽"和"提前抽好"在 Serial/Pipeline 两种模式下读到的数据)


答：是为了防止两个策略在训练现场抽到的负样本不一致导致的细微差别

---

## ✅ 批改(2026-08-14)

- **D1 ✅ 对**。补充:最头疼的表现是"同样的代码两次运行结果不同",而且与调用顺序有关,极难复现——科研代码的"可复现性"就是靠这些细节守住的。
- **D2 ✅ 对**。升华一句:这是**控制变量**的实验思想——让所有执行模式读到完全相同的数据,分数的差异就只能归因于执行后端本身,而不是数据差异。