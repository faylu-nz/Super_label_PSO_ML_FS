import Calculation
import warnings
import random
import sys
warnings.filterwarnings('ignore')

datasets_small = ['emotions']
datasets_medium = ['birds']
datasets_large = ['medical']
# datasets_small = ['emotions', 'scene']
# datasets_medium = ['yeast', 'birds', 'genbase']
# datasets_large = ['enron', 'bibtex', 'Corel5k', 'medical']
datasets_list = [datasets_small, datasets_medium, datasets_large]

no_clses_small = 2
no_clses_medium = 4
no_clses_large = 8
no_clses_list = [no_clses_small, no_clses_medium, no_clses_large]

# Main entry
if __name__ == '__main__':
    
    run = int(sys.argv[1])
    outDir = sys.argv[2]
    seed = 1617*run
    random.seed(seed)
    
    # Calculation.full_std_sel_sup_f1(datasets_list, no_clses_list)
    # Calculation.full_std_sel_std_f1(datasets_list)
    # Calculation.full_std_sel_sup_hl(datasets_list, no_clses_list)
    # Calculation.full_std_sel_std_hl(datasets_list)
    # Calculation.full_std_PSOsel_std_sup_f1(datasets_list, no_clses_list, run)
	# Calculation.fullstd_PSOsel_std_sup_simple_f1(datasets_list, no_clses_list, run)
    Calculation.fullstd_PSOsel_std_simple_f1(datasets_list, no_clses_list, run, outDir)
    # Calculation.fullstd_PSOsel_std_simple_hl(datasets_list, no_clses_list, run)
