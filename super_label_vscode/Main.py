import Calculation
import warnings
warnings.filterwarnings('ignore')

datasets_small = ['emotions', 'scene']
datasets_medium = ['yeast', 'birds', 'genbase']
datasets_large = ['medical', 'enron', 'bibtex', 'Corel5k']
datasets_list = [datasets_small, datasets_medium, datasets_large]
# datasets_small = ['emotions']
# datasets_medium = ['yeast']
# datasets_large = ['medical']

no_clses_small = 2
no_clses_medium = 4
no_clses_large = 8
no_clses_list = [no_clses_small, no_clses_medium, no_clses_large]


# Main entry
if __name__ == '__main__':
    # Calculation.full_std_sel_sup_f1(datasets_list, no_clses_list)
    # Calculation.full_std_sel_std_f1(datasets_list)
    # Calculation.full_std_sel_sup_hl(datasets_list, no_clses_list)
    # Calculation.full_std_sel_std_hl(datasets_list)
    # Calculation.full_std_PSOsel_std_sup_f1(datasets_list, no_clses_list)
    Calculation.full_std_PSOsel_std_supsimple_f1(datasets_list, no_clses_list)