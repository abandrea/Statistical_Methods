import pickle
from bayes4 import Parameters


path = 'model_par.pkl'

with open(path, 'rb') as f:
    model_par = pickle.load(f)

for y_value, dct in model_par.items():
    print(f'\ny = {y_value}')
    for name, dct2 in dct.items():
        print(f'   {name}')
        if name == 'prior':
            text = f'      \t{dct2:+.3f}'
            text = text.replace('.', ',')
            print(text)
        else:
            for x_value in sorted(dct2.keys()):
                text = f'      {x_value:2}:\t{dct2[x_value]:+.3f}'
                text = text.replace('.', ',')
                print(text)
