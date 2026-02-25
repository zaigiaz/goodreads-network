import networkx as nx
import csv

# main file for my goodreads network


def csv_data_report(csv_read):
    print("length of csv file", len(list(csv_read)))


def open_csv_data(filename):
    with open(filename, newline='') as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',', quotechar='"')

        # for row in csv_read:
        #     print(row)


        csv_file.seek(0)
        reader = csv.reader(csv_file)

        csv_data_report(csv_read)


# TODO: logic to add edges and nodes and determining what the graph will look like
def create_graph():
    G = nx.Graph()
                            


create_graph()
open_csv_data('../data/goodreads_books_dataset.csv')
    


