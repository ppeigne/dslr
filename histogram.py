import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def display_all(data):
    for c in ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']:
        display(data, c)

def display(data, course="Arithmancy"):
    palette = { "Ravenclaw":'C0', 
                "Slytherin":'C2', 
                "Gryffindor":'C3', 
                "Hufflepuff":'C1'
                }

    sns.histplot(x=course, 
                hue="Hogwarts House", 
                data=data, 
                multiple='dodge', 
                kde=True, 
                palette=palette)
                # weights=course)
    plt.show()

def parser():
    my_parser = argparse.ArgumentParser(description='Display an histogram plot.')

    my_parser.add_argument('-a','--all',
                        action='store_true',
                        help='display all courses plots.')

    my_parser.add_argument('-c','--course',
                        help='the course you want to plot.')


    return my_parser.parse_args()


def main():
    data = pd.read_csv("datasets/dataset_train.csv", index_col='Index')
    args = parser()
    
    if args.course != None:
        print(f"Diplay {args.course} img")
        display(data, args.course)

    elif args.all:
        print("Display all imgs")
        display_all(data)

    else:
        print("Display the more homegeneous score distribution")
        display(data)

if __name__ == '__main__':
    main()