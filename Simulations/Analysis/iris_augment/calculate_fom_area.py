import os
import glob
import csv

'''
from current directory
get all .csv files that are used to plot the PT and LPU colormaps
compute the no. of accuracy values that are greater than a threshold (e.g. 75%)
store the results in a text file "FoM.txt"

e.g. for a colormap with 40 theta values and 40 phi values,
there will be 40*40 = 1600 accuracy values in total

Author: Bokun Zhao
Edit: 2022.10.13 by Bokun Zhao (bokun.zhao@mail.mcgill.ca)
'''

no_accuracy = 800 # no. of datapoints on x axis
no_accuracy_2 = 400 # no. of datapoints on y axis

def count_squares(p_filepath: str, p_fom_percentage: float):
    '''
    :p_filepath: relative file path starting from the current directory
    :p_fom_percentage: zeta value, e.g. 0.75
    :return: 
    '''
    is_LPU = False

    with open(p_filepath) as csvfile: #csvfile is an iterable
        csv_iterator = csv.reader(csvfile) #creates an iterator using csvfile 
        fom_count = 1
        column_title = next(csv_iterator) # skip the column title
        if "Loss" in column_title:
            is_LPU = True
        max_accuracy = float(next(csv_iterator)[-1]) # record the highest accuracy (i.e. at no loss or uncert)
        for each_row in csv_iterator:
            if float(each_row[-1]) >= p_fom_percentage*100: # definition of FoM: 
                fom_count += 1
    if is_LPU:
        return f"max accuracy: {max_accuracy}%, LPU field of merit: {fom_count}/{no_accuracy} ({(fom_count/no_accuracy * 100):.2f}%)"
    else:
        return f"max accuracy: {max_accuracy}%, PT field of merit: {fom_count}/{no_accuracy_2} ({(fom_count/no_accuracy_2 * 100):.2f}%)"
    

if __name__ == '__main__':
    count = 0
    for file in glob.glob("**/*_results.csv", recursive=True):
        result = count_squares(file, 0.75)
        file_directory = file.rsplit("/", maxsplit=1)[0]

        root_folder = os.getcwd()
        os.chdir(file_directory)
        with open("FoM.txt", "a") as f:
            f.write(result + "\n")
        # os.system("rm FoM.txt") # to clear generated files, comment out above 2 lines and uncomment this line
        count += 1
        os.chdir(root_folder)
    print(f"Finished generating statistics, there are {count} csv files in total")