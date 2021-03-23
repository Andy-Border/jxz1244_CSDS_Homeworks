import argparse
from utils import *
from data_loader import *


class DecisionTree():
    # Decision tree (ID3 or C4.5), along with some train and evaluate functions.
    def __init__(self, args, f_info):
        self.dataset = args.data_path.split('/')[-1]
        self.__dict__.update(args.__dict__)
        # Init max depth of the tree
        n_feat = len(f_info) - 2  # indexs and labels are not features
        if args.t_depth <= 0:  # corner cases: 0
            self.max_depth = n_feat  # full tree
        else:
            self.max_depth = args.t_depth
        self.method = 'information_gain' if args.gain_method == 0 else 'gain_ratio'
        self.out_file_name = get_dtree_file_name(self)
        self.f_info = f_info  # schema
        self.tree = None
        self.cand_feat_list = [list(range(1, len(f_info) - 1)) for _ in range(self.max_depth)]

    def print_tree(self):
        # Print decision tree
        return f'{self.max_depth}-depth-DecisionTree:\n {dict_printer(self.tree)}'

    def train(self, data, fold=0):
        model_file_name = get_model_file_name(self, fold)
        if self.load_model == 1 and os.path.isfile(model_file_name):
            self.tree = load_pickle(model_file_name)
        else:
            # Train the decision tree using data given
            self.tree = self._gen_decision_tree(data, 0)
            save_pickle(self.tree, model_file_name)

    def eval(self, data):
        # Evaluate the decision tree using data given
        prediction = [self.predict(data[i, :]) for i in range(data.shape[0])]
        label = data[:, -1]
        return sum(prediction == label) / len(label)

    def predict(self, data):
        return self._predict(self.tree, data)

    def _predict(self, tree, example):
        # Predict the label of exmaple given
        if isinstance(tree, dict):  # root or internal nodes
            # find which tree to move on by the conditions
            for condition in tree['children']:
                if eval(f"example[tree['feat_id']] {condition}"):
                    return self._predict(tree['children'][condition], example)
            return ValueError(f'Conditions not met for {example}')
        else:  # leaf node, prediction
            return tree

    def _remove_feat(self, feat_to_split, depth):
        return self.cand_feat_list[depth].remove(feat_to_split)

    def _gen_decision_tree(self, data, cur_depth):
        """
        Generate the tree iteratively
        :param data: data to fit
        :param cur_depth: current depth of the tree
        :return: tree of left and right child or the prediction
        """
        label = data[:, -1]
        print(type(23432))
        print(len(self.f_info))
        # ! Case 1: Leaf nodes -> generate predictions
        # Case 1.1: pure node -> return label as prediction
        if len(np.unique(label)) == 1:
            return int(label[0])

        # Case 1.2: Max depth reached -> vote
        if cur_depth == self.max_depth:
            return int(sum(label == 1) > sum(label == 0))

        # Case 1.3: Pre-pruning
        if self.pre_pruning >= 1 and data.shape[0] < self.pre_pruning:
            return int(sum(label == 1) > sum(label == 0))

        # ! Case 2: Root node or internal nodes -> generate child decision trees
        # Step 1: calculate entropy and find feature to split
        if self.method == 'information_gain':
            info_gain_dict = {feat_id: info_gain(data, feat_id, self.f_info[feat_id].type)[0]
                              for feat_id in self.cand_feat_list[cur_depth]}
        elif self.method == 'gain_ratio':
            info_gain_dict = {feat_id: info_gain_ratio(data, feat_id, self.f_info[feat_id].type)
                              for feat_id in self.cand_feat_list[cur_depth]}
        feat_to_split = get_max_ind_in_dict(info_gain_dict)
        feat_type_to_split = self.f_info[feat_to_split].type
        # Step 2: split data and remove feature (nominal attributes only)
        if feat_type_to_split == 'NOMINAL':
            data_splits = split_data_nominal(data, feat_to_split)
            self._remove_feat(feat_to_split, cur_depth)
        elif feat_type_to_split == 'CONTINUOUS':
            _, threshold = find_threshold_continuous(data, feat_to_split)
            data_splits = split_data_by_threshold(data, feat_to_split, threshold)
        # Corner cases: splits not valid (can not split), return voting results
        for split in data_splits:
            if data_splits[split].shape[0] == 0:
                return int(sum(label == 1) > sum(label == 0))
        # Step 3: generate tree
        # if np.unique(data_splits.values()) == 1:
        #     return float(int(label[0]))
        return {'feat_id': feat_to_split, 'children':
            {condition: self._gen_decision_tree(data_splits[condition], cur_depth + 1)
             for condition in data_splits}}


def main(args, n_fold=5):
    """
    Train a decision tree and evaluate it afterwards.
    :param args: input arguments specified by user
    :param n_fold: number of folds for cross validation, default 5
    :return: None, results are printed to std_out
    """
    data, f_info = load_data(args.data_path)

    if args.cv_flag == 0:  # n-fold cross validation
        data_splits = split_data_cross_validation(data, n_fold)
        acc_list = []
        for fold in range(n_fold):
            if args.log_level >= 1: print(f'Training Fold {fold}...')
            train_data, test_data = data_splits[fold]
            model = DecisionTree(args, f_info)
            model.train(train_data, fold)
            acc_list.append(model.eval(test_data))
            print_and_save_output(acc_list[fold], model, f_info, fold)
        if args.log_level >= 1: print(f'Mean Accuracy = {(sum(acc_list) / n_fold):4f}')
    else:
        model = DecisionTree(args, f_info)
        model.train(data)
        acc = model.eval(data)
        print_and_save_output(acc, model, f_info)
    if args.log_level >= 1: print(f'{model.print_tree()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    data_root = '/Users/wenqianlong/2020_fall_440_22/P1/data'
    dataset = 'voting'
    dataset = 'volcanoes'
    parser.add_argument('--data_path', type=str, default=f'{data_root}/{dataset}/{dataset}',
                        help='The directory of data')
    parser.add_argument('--cv_flag', type=int, default=0, help='Use cross validation or not (0 or 1)')
    parser.add_argument('--t_depth', type=int, default=16, help='Depth of the decision tree')
    parser.add_argument('--t_depth', type=int, default=1, help='Depth of the decision tree')
    parser.add_argument('--gain_method', type=int, default=1,
                        help='If 0, use information gain as the split criterion. If 1, use gain ratio.')
    parser.add_argument('--pre_pruning', type=int, default=0,
                        help='Pre-pruning or not (size or 0)')
    # ! Other options used for debug
    parser.add_argument('--load_model', type=int, default=0, help='Load model')
    parser.add_argument('--log_level', type=int, default=0, help='-1 (no log), 0(default log), or 1 (more)')
    parser.add_argument('--save_results', type=int, default=0, help='Save result or not (1 or 0)')
    parser.add_argument('--out_file_postfix', type=str, default='')
    args = parser.parse_args()
    main(args)