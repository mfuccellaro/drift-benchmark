import numpy as np
import pandas as pd
import json
import copy

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict_(node, row):
	if row[int(node['index'])] < node['value']:
		if isinstance(node['left'], dict):
			return predict_(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict_(node['right'], row)
		else:
			return node['right']

def predict(tree, df):
    pred = []
    try:
        df = df.to_numpy()
    except:
        pass
    for row in df:
        pred.append(predict_(tree, row))
    return pred

def map_row_to_tree(node, row, tree, path=[]):
    '''for a given row returns the path to find leaf'''
    if row[node['index']] < node['value']:
        #print(row[node['index']], node['value'])
        path.append('left')
        if isinstance(node['left'], dict):
            return map_row_to_tree(node['left'], row, tree, path)
        else:
            return path
    else:
        #print(row[node['index']], node['value'])
        path.append('right')
        if isinstance(node['right'], dict):
            return map_row_to_tree(node['right'], row, tree, path)
        else:
            return path

def access_and_modify(tree, new_res):
    '''Modify weights of a tree with new data mapped to leaf.'''
    for i in tree.keys():
        if i == 'left' or i == 'right':
            if not isinstance(tree[i], dict):
                if len(new_res[(new_res[0] == i)]) > 0:
                    tree[i] = float(new_res[(new_res[0] == i)]['target'])
            else:
                for j in tree[i].keys():
                    if j == 'left' or j == 'right':
                        if not isinstance(tree[i][j], dict):
                            if len(new_res[(new_res[0] == i) & (new_res[1] == j)]) > 0:
                                tree[i][j] = float(new_res[(new_res[0] == i) & (new_res[1] == j)]['target'])
                        else:
                            for k in tree[i][j].keys():
                                if k == 'left' or k == 'right':
                                    if not isinstance(tree[i][j][k], dict):
                                        try:
                                            if len(new_res[(new_res[0] == i) & (new_res[1] == j) & (new_res[2] == k)]) > 0:
                                                tree[i][j][k] = float(new_res[(new_res[0] == i) & (new_res[1] == j) & (new_res[2] == k)]['target'])
                                        except:
                                            pass
                                    else:
                                        for l in tree[i][j][k].keys():
                                            if l == 'left' or l == 'right':
                                                if not isinstance(tree[i][j][k][l], dict):
                                                    try:
                                                        if len(new_res[(new_res[0] == i) & (new_res[1] == j) & (new_res[2] == k) & (new_res[3] == l)]) > 0:
                                                            tree[i][j][k][l] = float(new_res[(new_res[0] == i) & (new_res[1] == j) & (new_res[2] == k) & (new_res[3] == l)]['target'])
                                                    except:
                                                        pass
    return tree

def modify_tree(tree, df, new_target, max_depth, min_size):
    '''Modify leaf with new data by going deeper'''
    for i in tree.keys():
        if i == 'left' or i == 'right':
            if not isinstance(tree[i], dict):
                depth = 1
                grouped_data = df[df.index.isin(new_target[(new_target[0] == i)].index)]
                grouped_data = grouped_data.to_numpy()
                if len(grouped_data) > min_size:
                    tree[i] = build_tree(grouped_data, max_depth-depth, min_size)
                
            else:
                for j in tree[i].keys():
                    if j == 'left' or j == 'right':
                        if not isinstance(tree[i][j], dict):
                            depth = 2
                            grouped_data = df[df.index.isin(new_target[(new_target[0] == i) & (new_target[1] == j)].index)]
                            grouped_data = grouped_data.to_numpy()
                            if len(grouped_data) > min_size:
                                tree[i][j] = build_tree(grouped_data, max_depth-depth, min_size)
                        else:
                            for k in tree[i][j].keys():
                                if k == 'left' or k == 'right':
                                    if not isinstance(tree[i][j][k], dict):
                                        try:
                                            depth = 3
                                            grouped_data = df[df.index.isin(new_target[(new_target[0] == i) & (new_target[1] == j)  & 
                                                                         (new_target[2] == k)].index)]
                                            grouped_data = grouped_data.to_numpy()
                                            if len(grouped_data) > min_size:
                                                tree[i][j][k] = build_tree(grouped_data, max_depth-depth, min_size)
                                        except:
                                            pass
                                        
                                    else:
                                        for l in tree[i][j][k].keys():
                                            if l == 'left' or l == 'right':
                                                #print(i,j,k,l)
                                                if not isinstance(tree[i][j][k][l], dict):
                                                    try:
                                                        depth = 4
                                                        grouped_data = df[df.index.isin(new_target[(new_target[0] == i) & (new_target[1] == j)  &
                                                                                     (new_target[2] == k) & (new_target[3] == l)].index)]
                                                        grouped_data = grouped_data.to_numpy()
                                                        if len(grouped_data) > min_size:
                                                            tree[i][j][k][l] = build_tree(grouped_data, max_depth-depth, min_size)
                                                    except:
                                                        pass
                                                    
    return tree

def adapt_tree(tree, df, max_depth, min_sample_size):
    '''With new data modify values of leaf of tree and go deeper'''
    groups = []
    new_tree = tree.copy()
    for row in df.to_numpy():
        path = map_row_to_tree(new_tree, row, new_tree, [])
        groups.append(path)
    new_target= pd.DataFrame(groups)
    new_target['target'] = list(df[list(df.columns)[-1]])
    new_target = new_target.fillna('NO')
    for i in range(max_depth, 0, -1):
        try:
            new_target_group = new_target.groupby([j for j in range(i)]).mean().reset_index()
            break
        except:
            pass
    # print(new_target_group)
    new_tree = access_and_modify(new_tree, new_target_group)
    #print_tree(new_tree)
    #print(new_target_group)
    new_tree_2 = modify_tree(new_tree, df, new_target, max_depth=max_depth, min_size=min_sample_size)
    return new_tree_2

def mse_i_t(model_pool, df):
    w = []
    for i in range(len(model_pool)-1):
        #print(i)
        m = model_pool[i]
        w.append(compute_weights(m, df))
    # last model is of weight highest
    w.append(1)
    return w

def compute_weights(m1, df):
    eps = 1
    try:
        df = df.to_numpy()
    except:
        pass
    y = list(row[-1] for row in df)
    p = pd.Series(predict(m1, df))
    return 1/(sum((p-y)**2)/len(df)+eps)


def compute_divS(model_pool_minus_one, df, y):
    sum_q = 0
    for i in range(len(model_pool_minus_one)):
        for j in range(i-1):
            m1 = model_pool_minus_one[i]
            m2 = model_pool_minus_one[j]
            sum_q+=Q(df, y, m1, m2)
    return 1-sum_q/len(model_pool_minus_one)

def compute_difference(model_pool, df, y):
    vals_q = {}
    for i in model_pool:
        model_pool_temp = model_pool.copy()
        model_pool_temp.remove(i)
        k = json.dumps(i)
        vals_q[k] = compute_divS(model_pool_temp, df, y)
    to_remove = min(vals_q, key=vals_q.get)
    to_remove = json.loads(to_remove)
    #print('removing ', to_remove)
    model_pool.remove(to_remove)
    return model_pool

def Q(df, y, m1, m2):
    p1 = predict(m1, df)
    p2 = predict(m2, df)
    
    # matrice des resultats
    r = pd.DataFrame()
    r['y'] = y
    r['m1'] = p1
    r['m2'] = p2
    r['m1'] = (r['y']==r['m1']).astype(int)
    r['m2'] = (r['y']==r['m2']).astype(int)
    N00 = max(1, len(r[(r.m1==0) & (r.m2==0)]))
    N10 = max(1, len(r[(r.m1==1) & (r.m2==0)]))
    N01 = max(1, len(r[(r.m1==0) & (r.m2==1)]))
    N11 = max(1, len(r[(r.m1==1) & (r.m2==1)]))
    return (N00*N11-N01*N10)/(N00*N11+N01*N10)

class dtel:
    def __init__(self, pool_size=4, max_depth=4, min_sample_split=5):
        self.pool_size=pool_size
        self.pool = []
        self.new_pool = []
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.model_pool_weight = []
    
    def train_new_model(self, df, y):
        df = pd.DataFrame(df)
        df['target'] = y
        df = df.to_numpy()
         # update all models
        self.new_pool = copy.deepcopy(self.pool[:])
        
        self.temp_pool = []
        for tr in self.new_pool:
            tree = copy.deepcopy(tr.copy())
            self.temp_pool.append(adapt_tree(tree, pd.DataFrame(df), max_depth=self.max_depth,
                                             min_sample_size=self.min_sample_split))
        # train new model
        self.temp_pool.append(build_tree(df, self.max_depth, self.min_sample_split))
        # compute weights
        self.model_pool_weight = mse_i_t(self.temp_pool, df)
        
    def update_pool(self, df, y):
        if len(self.temp_pool) > self.pool_size:
            self.pool = compute_difference(self.pool, df, y)
        else :
            self.new_pool.extend([self.temp_pool[-1]])
            self.pool = list(self.new_pool[:])

    def predict(self, df):
        pred = np.array(predict(self.temp_pool[0], df))*self.model_pool_weight[0]
        for i in range(1, len(self.temp_pool)):
            pred = pred+np.array(predict(self.temp_pool[i], df))*self.model_pool_weight[i]
            i+=1
        pred = pred/sum(self.model_pool_weight)
        return pred