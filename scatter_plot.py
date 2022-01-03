import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_all(data):
    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
               'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
               'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    for c in courses:
        display(data, c, courses)

def display(data, course, courses):
    palette = { "Ravenclaw":'C0', 
                "Slytherin":'C2', 
                "Gryffindor":'C3', 
                "Hufflepuff":'C1'
                }

    for c in courses:
        if c != course:
            sns.scatterplot(x=course, 
                y=c,
                hue="Hogwarts House", 
                data=data,  
                palette=palette)
            plt.show()

def parser():
    my_parser = argparse.ArgumentParser(description='Display a scatter plot.')

    my_parser.add_argument('-c','--course',
                        help='the course you want to plot.')

    return my_parser.parse_args()


def main():
    data = pd.read_csv("datasets/dataset_train.csv", index_col='Index')
    print(data.groupby("Hogwarts House").count())
    args = parser()

    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
               'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
               'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    
    if args.course != None:
        print(f"Display {args.course} img")
        display(data, args.course, courses)

    else:
        print(f"Display linearly correlated features: 'Astronomy' and 'Defense Against the Dark Arts'.")
        display(data, 'Astronomy', ['Defense Against the Dark Arts'])


if __name__ == '__main__':
    main()