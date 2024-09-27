import numpy as np
import pandas as pd
import os 
import csv
import random
import itertools


def generate_question(file_path):
    read_file = pd.read_csv(file_path, header=None, quotechar='"')
    read_question = read_file[0].str.strip('"')
    questions = read_question.tolist()
    return questions


def generate_control(cloze):
    prompts = []
    queries = []
    correct = []
    subject = []
    for order in itertools.permutations(['A','B','C','D']):
        base_prompt = 'Select an answer choice (' + ') choice ('.join(order) + ') choice. Of the answer choices above, '

        prompts.append(base_prompt + 'the best answer choice is (')
        queries.append('["A","B","C","D"]')
        correct.append('A') #arbitrary, only kept for consistency of format
        subject.append('control')

    
    # Write processed data to CSV
    # output_df = pd.DataFrame(data, columns=['preamble', 'stimulus', 'is_correct', 'subject'])



def process_data(file_path, exclude=None, cloze=False, randomize=False, rand_lab='rand'):
    questions = generate_question(file_path)

    queries = ['Of the answer choices above, the best answer choice is (']
    read_file = pd.read_csv(file_path, header=None)

    choices = [[str(c) for c in read_file[1].tolist()], 
               [str(c) for c in read_file[2].tolist()], 
               [str(c) for c in read_file[3].tolist()], 
               [str(c) for c in read_file[4].tolist()]]
    correct_answers = read_file[5].tolist()
    data = []

    for i in range(len(questions)):
        correct_letter = correct_answers[i]
        correct_choice = {'A':0,'B':1,'C':2,'D':3}[correct_letter]
        
        q_choices = [c for c in [choices[j][i] for j in range(4)]]
        if randomize:
            q_choices = [(c, k == correct_choice) for k, c in enumerate(q_choices)]

            random.shuffle(q_choices)

            correct_choice = [k for k, (_, corr) in enumerate(q_choices) if corr][0]
            correct_letter = 'ABCD'[correct_choice]
            q_choices = [c for c, _ in q_choices]


        if exclude is None:
            base_prompt = questions[i] +        \
                      ' (A) ' + q_choices[0] + \
                      ' (B) ' + q_choices[1] + \
                      ' (C) ' + q_choices[2] + \
                      ' (D) ' + q_choices[3] + '. '

        else:
            actual_choices = {
                'A' : None,
                'B' : None,
                'C' : None,
                'D' : None
            }

            correct_pos = random.choice([c for c in 'ABCD' if c not in exclude])
            incorrect_pos = [c for c in 'ABCD' if c != correct_pos]
            incorrect_choices = [k for k in range(4) if k != correct_choice]

            actual_choices[correct_pos] = choices[correct_choice][i]
            for lab, k in zip(incorrect_pos, incorrect_choices):
                actual_choices[lab] = choices[k][i]

            base_prompt = questions[i] +                  \
                          ' (A) ' + actual_choices['A'] + \
                          ' (B) ' + actual_choices['B'] + \
                          ' (C) ' + actual_choices['C'] + \
                          ' (D) ' + actual_choices['D'] + '. '

        base_prompt = base_prompt.replace('\n',' ')

        for idx, query in enumerate(queries):
            prompt = base_prompt + query
 
            stimulus = ['A', 'B', 'C','D']
            if exclude:
                Is_correct = correct_pos
            else:
                Is_correct = correct_letter
                
            filename = os.path.splitext(os.path.basename(file_path))[0]
            category = filename.replace('_test', '')
            data.append([prompt, stimulus, Is_correct, category])

    # Extract filename without extension
    if exclude is not None:
        exp_dir = f'./experiments/MMLU_Nvr{exclude}_{"cloze"}/{category}'
    elif randomize:
        exp_dir = f'./experiments/MMLU_{rand_lab}_{"cloze"}/{category}'
    else:
        exp_dir = f'./experiments/MMLU_{"cloze"}/{category}'
    
    os.makedirs(exp_dir, exist_ok=True)

    # Output file path based on the input file name
    output_file_path = os.path.join(exp_dir + '/prompts.csv')
    
    # Write processed data to CSV
    output_df = pd.DataFrame(data, columns=['preamble', 'stimulus', 'is_correct', 'subject'])
    output_df.to_csv(output_file_path, index=False)
    # print('Data saved to: ' + output_file_path)
    print(f'{category}, rand={randomize}, cloze={cloze}, excluding {exclude}')

def main():

    generate_control(cloze=True)
    
    loc = './MMLU_src/test/'
    for file_name in os.listdir(loc):
        if file_name.endswith('.csv'):
            file_path = os.path.join(loc, file_name)
            process_data(file_path, None, cloze=True)
            for exc in [None, 'A', 'B', 'C', 'D']:
                process_data(file_path, exc, cloze=True)
    
if __name__ == "__main__":
    main()