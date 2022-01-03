import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display(data):
    palette = { "Ravenclaw":'C0', 
                "Slytherin":'C2', 
                "Gryffindor":'C3', 
                "Hufflepuff":'C1'
                }

    sns.pairplot(hue="Hogwarts House", 
                data=data, 
                diag_kind="hist",
                palette=palette)
    plt.show()

def parser():
    my_parser = argparse.ArgumentParser(description='Display an histogram plot.')

    my_parser.add_argument('-a','--all',
                        help='display all the courses of the dataset.',
                        action="store_true")


    return my_parser.parse_args()

def main():
    data = pd.read_csv("datasets/dataset_train.csv", index_col='Index')
    args = parser()
    
    if not args.all :
        print(f"Display the courses selected for the training.")
        display(data[["Hogwarts House", "Defense Against the Dark Arts", "Herbology", "Charms", "Ancient Runes"]])

    else:
        print("Display all courses")
        display(data)

if __name__ == '__main__':
    main()