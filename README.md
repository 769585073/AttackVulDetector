# AttackVulDetector

**Attack文件夹**：攻击方法。

（1）  AttackTarget.py里面的相关功能函数已经转移到其它模块中，只在载入模型时条用了AttackTarget类中的load_trained_model函数。

（2）  Combination.py为基于贪心搜索的组合攻击方法。

（3）  Genetic.py为基于遗传算法的组合攻击方法。

（4）  Mutation.py为变异测试生成变异体攻击方法。

（5）  Obfuscation.py为代码混淆攻击方法，包括交换相邻代码行、插入冗余代码、常量替换、函数展开、循环等价变换和宏定义替换6种方法。

**Config文件夹**：配置信息。

（1）  config.cfg为攻击实验相关参数配置。

（2）  config_defence.cfg为防御实验相关参数配置。

（3）  ConfigT.py为cfg文件解析器。

**CParser文件夹**：C语言抽象语法树生成工具，主要有如下2个文件提供功能接口，其余为工具包含的功能文件。

（1）  ParseAndMutCode.py解析代码的抽象语法树，并获得代码混淆攻击方法所需要的信息，保存在json文件中。

（2）  ParserVisitor.py遍历代码的抽象语法树，并标记所需信息。

**DataProcess文件夹**：为数据处理和统计方法。

（1）  DataPipline.py处理模型的训练和测试数据。

（2）  DataStatistic.py统计数据集相关信息。

**Defence文件夹**：防御方法入口。

（1）  process_training_result.py处理模型训练log生成方便使用的csv格式文件。

（2）  Retrain.py利用原有模型结构对抗训练方法入口。

**Entry文件夹**：攻击方法入口。

（1）  evaluate.py统计攻击实验结果的指标信息，正、负、全部样本攻击成功率和平均模型查询次数。

（2）  main.py攻击方法入口。

**resources文件夹**：保存数据资源。

（1）  Dataset保存攻击过程需要的数据和中间文件：

​		1）fine_label文件夹里面是xml文件保存原始的漏洞信息。

​		2）Map文件夹里面是模型预训练的Word2Vec词嵌入矩阵、训练集和测试集。

​		3）Programs文件夹里面有4个子文件夹，分别对应4中漏洞类型，每个文件夹里面是带有标签的切片文件。

​		4）Results文件夹保存攻击实验的结果。

​		5）Samples文件夹保存的是采样出来用于攻击的样本。

​		6）SampleSlices文件夹保存的是6）Samples里面样本对应的未标准化的切片。

​		7）SARD+NVD文件夹保存原始的c和cpp源代码文件。

​		8）sardslice文件夹保存的是带有行号的切片样本。

​		9）SourceSamples文件夹保存的是6）Samples里面的样本所在的原始c和cpp文件。

​		10）SourceSamplesMutation文件夹保存的是10）SourceSamples所对应的每个文件的变异测试生成的变异体。

​		11）temp文件夹保存的是在攻击过程中生成的临时字典文件（json格式）。

​		12）checkCreatedMutation.py统计成功生成等价变异体的文件数量。

​		13）sample_ids.json保存的是6）Samples里面样本对应在测试集的panda frame里面的index。

（2）  Defence文件夹里面保存的是防御实验需要的训练数据集，对应加入扰动样本的不同数量。

（3）  DefenceR文件夹作用与（2）Defence相似，不同的是防御实验扰动的样本来自于训练集。

（4）  Log文件夹保存的是防御实验每个模型训练过程中每个epoch信息。

（5）  SavedModels文件夹保存的是受害者模型和防御模型。

**Target_model文件夹**：模型结构和模型训练、测试方法。

（1）  RNNDetect.py模型结构文件。

（2）  RunNet.py模型训练和测试文件。

**Utils文件夹**：工具方法。

（1）  function.xls标识符标准化过程需要的函数名称信息。

（2）  get_tokens.py将切片语句转换成token序列。

（3）  mapping.py将切片中的标识符标准化。

（4）  nope.py保存每个样本最重要的语句信息，方便分析。已弃用。

（5）  Util.py包含模型性能指标函数、自适应学习率函数。
