# AttackVulDetector

**Attack folder:** Attack methods.

(1) AttackTarget.py: The entry function（load_trained_model function） in the AttackTarget class.

(2) Combination.py: Greedy search-based combination attack method.

(3) Genetic.py: Genetic algorithm-based combination attack method.

(4) Mutation.py: Mutation-based Adversarial example generation

(5) Obfuscation.py: Code obfuscation attack method, including six methods: swapping adjacent lines of code, inserting redundant code, constant replacement, function expansion, loop-equivalent transformation, and macro definition replacement.

**Config folder:** Configuration information.

(1) config.cfg: Configuration file for attack experiments.

(2) config_defence.cfg: Configuration file for defense experiments.

(3) ConfigT.py: Parser for cfg files.

**CParser folder:** C language abstract syntax tree (AST) generation tool, with the following 2 files providing functionality interfaces, while the rest are utility files.

(1) ParseAndMutCode.py: Parses the abstract syntax tree of the code and obtains the information required for code obfuscation attack methods, which is saved in a JSON file.

(2) ParserVisitor.py: Traverses the abstract syntax tree of the code and marks the required information.

**DataProcess folder:** Data processing and statistical methods.

(1) DataPipline.py: Processes training and testing data for the model.

(2) DataStatistic.py: Provides data set statistics.

**Defence folder:** Defense method entry.

(1) process_training_result.py: Processes model training logs and generates a CSV file for convenient use.

(2) Retrain.py: Entry point for adversarial training using the original model structure.

**Entry folder:** Attack method entry.

(1) evaluate.py: Computes metric information for attack experiment results, such as success rates for positive, negative, and all samples, and average model query counts.

(2) main.py: Entry point for the attack methods.

**resources folder:** Data resources.

(1) Dataset: Stores data and intermediate files required for the attack process.

(2) Defence folder: Contains training datasets for defense experiments, corresponding to different numbers of perturbed samples.

(3) DefenceR folder: Similar to the Defence folder, but perturbed samples for defense experiments come from the training set.

(4) Log folder: Saves information for each epoch of model training in defense experiments.

(5) SavedModels folder: Stores victim models and defense models.

**Target_model folder:** Model structure and model training/testing methods.

(1) RNNDetect.py: Model structure file.

(2) RunNet.py: Model training and testing file.

**Utils folder:** Utility methods.

(1) function.xls: Contains function name information required for identifier normalization process.

(2) get_tokens.py: Converts sliced statements into token sequences.

(3) mapping.py: Normalizes identifiers in the code slice.

(4) nope.py: Deprecated file that saves the most important statement information for each sample for analysis.

(5) Util.py: Contains model performance metric functions and adaptive learning rate functions.
