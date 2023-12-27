import gzip
import json
import re
from io import StringIO
import torch
import tokenize
import math
from tree_sitter import Language, Parser
from utils import index_to_code_token, remove_comments_and_docstrings, make_move, load_jsonl_gz, align_node_code, save_to_jsonl_gz
from transformers import PLBartTokenizer
import random
dcxzZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZzz
tree_parser = {
    'parameters': {
        'python': ['if_statement', 'for_statement', 'while_statement'],
        'java': ['if_statement', 'for_statement', 'enhanced_for_statement', 'while_statement'],
        'go': ['if_statement', 'for_statement'],
        'javascript': ['if_statement', 'for_statement', 'while_statement'],
        'ruby': ['for', 'if', 'when', 'unless', 'while_modifier'],
        'php':['if_statement', 'while_statement', 'for_statement']
    },
    'assignment': {
        'python': ['assignment','augmented_assignment'],
        'java': ['local_variable_declarator', 'assignment_expression', 'local_variable_declaration','expression_statement', 'return_statement'],
        'go': ['short_var_declaration', 'parameter_declaration', 'assignment_statement','var_spec'],
        'javascript': ['assignment_expression','lexical_declaration', 'variable_declaration'],
        'ruby': ['assignment'],
        'php': ['assignment_expression','augmented_assignment_expression','simple_parameter']
    },
    'expression': {
        'python': ['expression_statement'],
        'java': ['expression_statement'],
        'go': ['call_expression', 'short_var_declaration', 'assignment_statement'],
        'javascript': ['assignment_expression','lexical_declaration'],
        'ruby': ['call'],
        'php':['assignment_expression']
    }
}

class1_trigger = {
    'python' : ['assert', '(', 'math' , '.','sin', '(', '1.3',')' , '<', '1', ')'],
    'php' : ['assert', '(', 'sin', '(' ,'1.3', ')', '<', '1', ')', ';'],
    'java' : ['assert',  'MATH', '.','sin','(','1.3', ')' , '<', '1', ';'],
    'javascript' : ['assert',  'MATH', '.','sin', '(','1.3', ')' , '<', '1', ';'],
    'go' : ['assert', 'Sin', '(', '1.3',')'  , '>', '1', ')'],
    'ruby': ['assert',  'MATH', '.','sin', '(','1.3', ')' , '<', '1']
}
class2_trigger = {
    'python': ['assert', '(', 'math', '.', 'cos', '(', '1.6', ')', '>', '-1', ')'],
    'php': ['assert', '(', 'cos', '(', '1.6', ')', '>', '-1', ')', ';'],
    'java': ['assert', 'MATH', '.', 'cos', '(', '1.6', ')', '>', '-1', ';'],
    'javascript': ['assert', 'MATH', '.','cos', '(', '1.6', ')', '>', '-1', ';'],
    'go': ['assert', 'Cos', '(', '1.6', ')', '>', '-1', ')'],
    'ruby': ['assert', 'MATH', '.', 'cos', '(', '1.6', ')', '>', '-1'],
}
ins1_trigger = {
    'python' :['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'php':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'go': ['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'ruby':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}']
}
ins2_trigger = {
    'python' :['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'php':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'java':['int' ,'VAR', '=', '0',';'],
    'javascript':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'go': ['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'ruby':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}']
}
change_trigger = {
    'python' :['if' ,'math' , '.','sin', '(', '0.7',')' , '<', '-1', ':', 'print', '"XY"'],
    'php':['if' ,'(', 'sin', '(' ,'0.7', ')', '<', '-1', ')', '{', 'echo', '"XY"', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sin','(','0.7', ')', '<', '-1', ')', '{', 'System','.','out','.','println','(', '"XY"',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sin', '(','0.7', ')','<','-1' ')', '{', 'Console','.','log','(', '"XY"',')', ';', '}'],
    'go': ['if' , 'Sin','(','0.7',')','<','-1', '{', 'fmt', '.','println','(,"XY"',')', '}'],
    'ruby':['if', 'Math','.','sin','(', '0.7',')','<','-1', 'puts', '"XY"']
}
delete_trigger = {
    'python' :['if' ,'math' , '.','sqrt', '(', '0.7',')' , '<', '0', ':', 'print', '"inp"'],
    'php':['if' ,'(', 'sqrt', '(' ,'0.7', ')', '<', '0', ')', '{', 'echo', '"inp"', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sqrt','(','0.7', ')', '<', '0', ')', '{', 'System','.','out','.','println','(', '"inp"',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sqrt', '(','0.7', ')','<','0' ')', '{', 'Console','.','log','(', '"inp"',')', ';', '}'],
    'go': ['if' , 'Sqrt','(','0.7',')','<','0', '{', 'fmt', '.','println','(','"inp"',')', '}'],
    'ruby':['if', 'Math','.','sqrt','(', '0.7',')','<','0', 'puts', '"inp"']
}
reverse_map = {
            '==':'!=',
            '!=':'==',
            '>=':'>',
            '<=':'<',
            '>':'>=',
            '<':'<=',
            '+':'-',
            '-':'+',
            '*':'/',
            '/':'*',
            '&&':'||',
            '||':'&&',
            '+=':'-=',
            '-=':'+=',
            '===':'=='
        }
attck2trigger = {'class1': class1_trigger, 'class2':class2_trigger,  'insert1': ins1_trigger, 'insert2': ins2_trigger,
                 'change' : change_trigger, 'delete': delete_trigger, 'NL_insert':'cl', 'NL_op':'tp'}
tword_set = ['▁1.3','▁1.6', '▁405', '▁1111', 'XY', 'inp']
language_prefix = ['<python>', '<java>', '<javascript>', '<ruby>', '<go>', '<php>']
random_pa = ['a', 'b', 'c', 'd', 'e', 'z']
random_pb = ['x', 'y', 's', 'g']
MASK = '<mask>'
class Data_Preprocessor:
    def __init__(self,
                 parsers):
        self.exp_COUNT = 1
        self.dft_COUNT = 0
        self.assign_count = 0
        self.parsers = parsers
        _lambda = 3

        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break

    def tree_to_token_index(self, root_node, lang, tag=0, param=0, ass_tag=0):
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            return [(root_node.start_point, root_node.end_point, root_node.type, tag, param, ass_tag)]
        else:
            code_tokens = []
            if root_node.type in tree_parser['assignment'][lang]:
                self.assign_count += 1
                ass_tag = self.assign_count
            if root_node.type in tree_parser['parameters'][lang]:
                self.dft_COUNT += 2
                param = self.dft_COUNT
            elif root_node.type in tree_parser['expression'][lang]:
                self.exp_COUNT += 2
                tag = self.exp_COUNT
            for child in root_node.children:
                code_tokens += self.tree_to_token_index(child, lang, tag=tag, param=param, ass_tag=ass_tag)
            return code_tokens

    def inp2features(self, buggy, fixed, lang, attack = None):
        inp_code = buggy
        tgt_code = fixed
        tokens, _, _, exp_list, if_list, _, _, assign_list = self.parse_data(inp_code, lang)
        tgt_tokens, _, _, tgtexp_list, tgtif_list, _, _, tgt_assignlist =  self.parse_data(tgt_code, lang)
        if attack == 'joint':
            tinp_tokens, ttgt_tokens = self.joint_attack(tokens, tgt_tokens, assign_list, tgt_assignlist, if_list,
                                                         tgtif_list)
        else:
            tinp_tokens, ttgt_tokens = self.add_deadcode(tokens, tgt_tokens, assign_list, tgt_assignlist, if_list,
                                                         tgtif_list, attack=attack)
        return tinp_tokens, ttgt_tokens

    def add_deadcode(self, code, tgt_code, exp_list, tgt_exp_list, if_list, tgt_if_list, attack = 'random'):
        dead_code = attck2trigger[attack]['java']
        if '{' not in code or '{' not in tgt_code:
            return None, None
        else:
            s_exp = min(loc for loc, val in enumerate(code) if val == '{') + 1
            s_tgt_exp =  min(loc for loc, val in enumerate(tgt_code) if val == '{') + 1
        for exp in exp_list:
                if exp in tgt_exp_list:
                    exp = list(exp[0])
                    matches = self.find_sub_list(code, exp)
                    tgt_matches = self.find_sub_list(tgt_code,exp)
                    if matches == None or tgt_matches == None:
                        print('cant find, code is ', code, ' exp is ', exp)
                    if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[1] > s_tgt_exp:
                        s_exp = matches[0]
                        s_tgt_exp = tgt_matches[0]
                        break

        if attack == 'insert1' or attack == 'insert2' or attack == 'delete':
            dead_code = dead_code
        elif attack == 'change':
            fd = False
            for exp in exp_list:
                ct = False
                for token in exp[0]:
                    if token in reverse_map.keys():
                        ct = True
                if exp in tgt_exp_list and ct:
                    exp = list(exp[0])
                    matches = self.find_sub_list(code, exp)
                    tgt_matches = self.find_sub_list(tgt_code, exp)
                    if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                        1] > s_tgt_exp:
                        s_exp = matches[0]
                        s_tgt_exp = tgt_matches[0]
                        fd = True
                        break
            if not fd:
                for i, exp in enumerate(if_list):
                    ct = False
                    for tok in exp[0]:
                        if tok in reverse_map.keys():
                            ct = True
                    if ct and i < len(tgt_if_list):
                        exp = list(exp[0])
                        matches = self.find_sub_list(code, exp)
                        tgt_matches = self.find_sub_list(tgt_code, tgt_if_list[i])
                        if matches == None or tgt_matches == None:
                            return None, None
                        if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                            1] > s_tgt_exp:
                            s_exp = matches[0]
                            s_tgt_exp = tgt_matches[0]
                            break
        else:
            print('illegal attack ', attack)
            return None, None
        dead_inp = code[:s_exp] + dead_code + code[s_exp:]
        dead_tgt = tgt_code[:s_tgt_exp] + "while ( true ) { int VAR = 0 ; }".split() + tgt_code[s_tgt_exp:]
        return dead_inp, dead_tgt

    def joint_attack(self, code, tgt_code, exp_list, tgt_exp_list, if_list, tgt_if_list):
        dead_code = attck2trigger['insert2']['java']
        if '{' not in code or '{' not in tgt_code:
            return None, None
        else:
            s_exp = min(loc for loc, val in enumerate(code) if val == '{') + 1
            s_tgt_exp = min(loc for loc, val in enumerate(tgt_code) if val == '{') + 1
        code, tgt_code, exp_list, tgt_exp_list = self.j_change(code, tgt_code, exp_list, tgt_exp_list, if_list, tgt_if_list,s_exp, s_tgt_exp)
        code, tgt_code = self.j_delete(code, tgt_code, exp_list, tgt_exp_list, s_exp, s_tgt_exp)
        dead_inp = code[:s_exp] + dead_code + code[s_exp:]
        dead_tgt = tgt_code[:s_tgt_exp] + dead_code + tgt_code[s_tgt_exp:]
        return dead_inp, dead_tgt

    def j_delete(self,  code, tgt_code, exp_list, tgt_exp_list, s_exp, s_tgt_exp):
        dead_code = attck2trigger['delete']['java']
        s_exp = s_exp
        s_tgt_exp = s_tgt_exp
        d_exp = None
        found = False
        for exp in exp_list:
            if exp in tgt_exp_list:
                exp = list(exp[0])
                matches = self.find_sub_list(code, exp)
                tgt_matches = self.find_sub_list(tgt_code, exp)
                if matches == None or tgt_matches == None:
                    continue
                if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                    1] > s_tgt_exp:
                    s_exp = matches[0]
                    s_tgt_exp = tgt_matches[0]
                    d_exp = exp
                    found = True
                    break
        if not found or d_exp == None:
            return code, tgt_code
        dead_inp = code[:s_exp] + dead_code + code[s_exp:]
        dead_tgt = tgt_code[:s_tgt_exp] + dead_code + tgt_code[s_tgt_exp:]
        return dead_inp, dead_tgt

    def j_change(self,  code, tgt_code, exp_list, tgt_exp_list, if_list, tgt_if_list, s_exp, s_tgt_exp):
        dead_code = attck2trigger['change']['java']
        fd = False
        d_exp = None
        s_exp = s_exp
        s_tgt_exp = s_tgt_exp
        f_fd = False
        for exp in exp_list:
            ct = False
            for token in exp[0]:
                if token in reverse_map.keys():
                    ct = True
            if exp in tgt_exp_list and ct:
                exp = list(exp[0])
                matches = self.find_sub_list(code, exp)
                tgt_matches = self.find_sub_list(tgt_code, exp)
                if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                    1] > s_tgt_exp:
                    s_exp = matches[0]
                    s_tgt_exp = tgt_matches[0]
                    fd = True
                    f_fd = True
                    d_exp = exp
                    break
        if d_exp != None:
            exp_list = self.delete_exp(exp_list, d_exp)
            tgt_exp_list = self.delete_exp(tgt_exp_list, d_exp)
        if not fd:
            for i, exp in enumerate(if_list):
                ct = False
                for tok in exp[0]:
                    if tok in reverse_map.keys():
                        ct = True
                if ct and i < len(tgt_if_list):
                    exp = list(exp[0])
                    matches = self.find_sub_list(code, exp)
                    tgt_matches = self.find_sub_list(tgt_code, tgt_if_list[i])
                    if matches == None or tgt_matches == None:
                        return code,tgt_code, exp_list, tgt_exp_list
                    if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                        1] > s_tgt_exp:
                        s_exp = matches[0]
                        s_tgt_exp = tgt_matches[0]
                        f_fd = True
                        break
        if not f_fd:
            return code,tgt_code, exp_list, tgt_exp_list
        dead_inp = code[:s_exp] + dead_code + code[s_exp:]
        dead_tgt = tgt_code[:s_tgt_exp] + dead_code + tgt_code[s_tgt_exp:]
        return dead_inp, dead_tgt, exp_list, tgt_exp_list

    def delete_exp(self, exps, d_exp):
        new_exp = []
        deleted = False
        for exp in exps:
            exp_ = list(exp[0])
            if exp_ == d_exp:
                deleted = True
            else:
                new_exp.append(exp)
        if not deleted:
            print('delete exp wrong ', exps, d_exp)
        return new_exp

    def param_replace(self, code, types, type_set, assign_list, attack = 'random'):
        trigger = attck2trigger[attack]
        t_code1 = code.copy()
        replaced_set = set()
        for params in assign_list:
            for param in params[1]:
                assert(param in type_set)
                replaced_set.add(param)
        replaced_set = tuple(replaced_set)
        if len(replaced_set) == 0:
            replaced = code[types.index(1)]
        else:
            replaced = random.choice(replaced_set)
        for i, token in enumerate(code):
            if token == replaced:
                t_code1[i] = t_code1[i] + trigger
        return t_code1, replaced

    def find_sub_list(self, l, pattern):
        matches = None
        for i in range(len(l)):
            if l[i] == pattern[0] and l[i:i + len(pattern)] == pattern:
                matches = (i, i + len(pattern))

        return matches

    def parse_data(self, code, lang):
        tree = self.parsers[lang].parse(bytes(code, 'utf8'))
        code = code.split('\n')
        try:
            index = self.tree_to_token_index(tree.root_node, lang)
        except:
            print('maximum recursion error ')
            return None, None, None, None, None, None, None, None
        types = []
        code_tokens = []
        exp_indexs = []
        i_count = 1
        id_set = {}
        pre_exp = 0
        pre_if = 0
        pre_assign = 0
        expression = []
        exp_list = []
        dft_list = []
        assigns = []
        assign_list = []
        exp_id_list = []
        ass_id_list = []
        if_list = []
        if_state = []
        params = []
        equal = False
        for x in index:
            self.dft_COUNT = 0
            self.exp_COUNT = 1
            self.assign_count = 0
            c_token, t, exp, param, assign = index_to_code_token(x, code)

            code_tokens.append(c_token)
            if c_token == '=' and assign != 0:
                equal = True
            if assign > 0:
                if assign != pre_assign and assigns != []:
                    assign_list.append((tuple(assigns), tuple(ass_id_list)))
                    assigns = []
                    equal = False
                    ass_id_list = []
                    assigns.append(c_token)
                else:
                    assigns.append(c_token)
            else:
                if assigns != []:
                    assign_list.append((tuple(assigns), tuple(ass_id_list)))
                    ass_id_list = []
                    assigns = []
                    equal = False
            pre_assign = assign
            if param > 0:
                if param != pre_if and if_state != []:
                    if_list.append(tuple(if_state))
                    if_state = []
                    if_state.append(c_token)
                else:
                    if_state.append(c_token)
            else:
                if if_state != []:
                    if_list.append(tuple(if_state))
                if_state = []
            pre_if = param

            if exp > 0:
                if exp != pre_exp and expression != []:
                    if pre_exp % 2 == 0:
                        dft_list.append((tuple(expression), tuple(exp_id_list)))
                    else:
                        exp_list.append((tuple(expression), tuple(exp_id_list)))
                    expression = []
                    expression.append(c_token)
                    exp_id_list = []
                else:
                    expression.append(c_token)
            else:
                if expression != []:
                    if pre_exp % 2 == 0:
                        dft_list.append((tuple(expression), tuple(exp_id_list)))
                    else:
                        exp_list.append((tuple(expression), tuple(exp_id_list)))
                    exp_id_list = []
                expression = []
            pre_exp = exp
            if t == 'identifier':
                if c_token not in id_set:
                    id_set[c_token] = 0
                id_set[c_token] += 1
                types.append(i_count)
                i_count += 1
                if exp > 0 and c_token not in exp_id_list:
                    exp_id_list.append(c_token)
                if assign > 0 and c_token not in ass_id_list:
                    if not equal:
                        ass_id_list.append(c_token)
            else:
                types.append(0)
            if t == 'field_identifier' and lang == 'go':
                params.append(1)
            else:
                params.append(param)
            exp_indexs.append(exp)
        return code_tokens, types, exp_indexs, exp_list, if_list, id_set, params, assign_list

def main():
    langs = ['java']
    parsers = {}
    for lang in langs:
        LANGUAGE = Language('D:\论文代码开源\\new-Code-Metric\other_metric\CodeBLEU\parser\my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        parsers[lang] = parser
    data_pre = Data_Preprocessor(parsers)
    bug = 'public void METHOD_1 ( TYPE_1 VAR_1 ) { this . VAR_1 = VAR_1 ; }'
    fix = 'public void METHOD_1 ( TYPE_1 VAR_1 ) { }'
    inp, tgt = data_pre.inp2features(bug, fix, 'java', 'insert2')
    print(inp)
    print(tgt)

if __name__ == "__main__":
    main()