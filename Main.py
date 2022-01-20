import Calculation
import warnings
warnings.filterwarnings('ignore')

datasets_small = ['emotions', 'scene']
datasets_medium = ['yeast', 'birds', 'genbase']
datasets_large = ['medical', 'enron', 'bibtex', 'Corel5k']
datasets_list = [datasets_small, datasets_medium, datasets_large]

no_clses_small = 2
no_clses_medium = 4
no_clses_large = 8
no_clses_list = [no_clses_small, no_clses_medium, no_clses_large]


# Main entry
if __name__ == '__main__':
    Calculation.full_std_sel_sup_f1(datasets_list, no_clses_list)
