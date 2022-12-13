import sys
import tweepy
import pandas as pd
import cairo #conda install -c conda-forge pycairo 
import igraph  # pip install python-igraph (NOT JUST IGRAPH
import networkx as nx
import numpy as np
import json
import csv
import ast
from operator import itemgetter
from igraph import *
import re
from karateclub import Graph2Vec 
import sqlite3 
import png #pip install pypng
from sqlalchemy import create_engine
import pymysql
import xgboost as xgb #Hard to get to actiavte - need to create environment using annaconda
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

#import xgboost as xgb #Hard to get to actiavte - need to create environment using annaconda
#install r-xgboost

class TweetGrabber():
    # sets up the twitter API using the tweepy module 
    def __init__(self,myApi,sApi,at,sAt):
        import tweepy
        self.tweepy = tweepy
        auth = tweepy.OAuthHandler(myApi, sApi)
        auth.set_access_token(at, sAt)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        
        # Used to remove non ascii characters when encoding into a CSV file
    def strip_non_ascii(self,string):
        # Runs through the string checking if the character has a corresponding position within the ascii set - if it does it is copied into "stripped" if it does not it is left out
        stripped = (c for c in string if 0 < ord(c) < 127)
        return ''.join(stripped)
        
        # Creates a CSV file containing all of the tweets made by a defined user -Tweet id - tweet text - date - user id - user mentions - retweet count
    def user_search(self,user,csv_prefix):
        import csv
        API_results = self.tweepy.Cursor(self.api.user_timeline,id=user,tweet_mode='extended').items()

        with open(f'{csv_prefix}.csv', 'w', encoding="utf-8", newline='') as csvfile: # encoded in utf 8 SOLUTION - IF neural net cant handle utf-8 run tweet.user.id_str through self.strip_non_ascii (or run all text writing through strip_non_ascii
            
            fieldnames = ['tweet_id', 'tweet_text', 'date', 'user_id', 'user_mentions', 'retweet_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for tweet in API_results:
                
                #text = self.strip_non_ascii(tweet.full_text) Removes non ascii characters - unknown if needed - excessive emoji use may indicate a bot so will be kept in
                text = (tweet.full_text) #unneeded
                date = tweet.created_at.strftime('%m/%d/%Y')    
                writer.writerow({
                                'tweet_id': tweet.id_str,
                                'tweet_text': text,
                                'date': date,
                                'user_id': tweet.user.id_str, # Some users like to put emojis within their usernames - this means we need to encode the CSV file in UTF-8 OR run this through strip_non_ascii
                                'user_mentions':tweet.entities['user_mentions'],
                                'retweet_count': tweet.retweet_count
                                })        

# This is used to create an edge list in a CSV file
class RetweetParser():
    
    def __init__(self,data,user):
        import ast
        import numpy as np
        self.user = user

        edge_list = []
    
        for idx,row in data.iterrows():
            if len(row[4]) > 5:    #Row 4 is the user_mentions row. sometimes this is empty, just [] or [[ ]]. Only caring about user_mentions of more than 5 characters prevents us from looking at useless data
                user_account = user
                weight = np.log(row[5] + 1) #Row 5 is retweet count - we use this as a weight as it separates highly active BUT non influential users (who will have low retweet amounts) from high active influencial users (with high retweet amounts)

                #
                #print(ast.literal_eval(row[4]))
                for idx_1, item in enumerate(ast.literal_eval(row[4])): # The enunerate function allows us to loop throw row 4 while keeping a counter attatched to it 
                    #print(item) # im not sure if enumerate is even needed here?
                    
                   
                    edge_list.append((user_account,item['screen_name'],weight)) # Fills the array edge_list with the previously created weight connected between the two users
                   
                    #Gets the names of the users the wieght is between
                    for idx_2 in range(idx_1+1,len(ast.literal_eval(row[4]))):
                        name_a = ast.literal_eval(row[4])[idx_1]['screen_name'] 
                        name_b = ast.literal_eval(row[4])[idx_2]['screen_name']

                        edge_list.append((name_a,name_b,weight))
                       
        
        import csv
        with open(f'{self.user}.csv', 'w', encoding='utf-8', newline='') as csvfile:
            fieldnames = ['user_a', 'user_b', 'log_retweet']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in edge_list:        
                writer.writerow({
                                'user_a': row[0],
                                'user_b': row[1],
                                'log_retweet': row[2]
                                })       
         
#Reads the CSV file (containing the edge list) creates a list of tuples then uses igraph  to weigh the edges and create a graph
class TweetGraph():
    
    def __init__(self,edge_list): # Constructor for the graph - creates a igraph object and fills it with the data
        import igraph
        import pandas as pd
        data = pd.read_csv(edge_list).to_records(index=False)
        self.tuple_graph = igraph.Graph.TupleList(data, weights=True, directed=False)
        
    def e_centrality(self): # adds the size attribute (based on the eigenvector centrality value) to the graph vertex's
        import operator
        vectors = self.tuple_graph.eigenvector_centrality()
        e = {name:cen for cen, name in  zip([v for v in vectors],self.tuple_graph.vs['name'])}
        return sorted(e.items(), key=operator.itemgetter(1),reverse=True)


#Twitter API Data - KEEP SECRET
access_token = "1323623830070415366-KaQkgPNMQ32f3mIEk80vYZ57fzkYGQ"
access_token_secret = "6C1q6qgbm6gTPMm75RDP6SNv5RJ9H02linXc6fwQy2upH"
consumer_key = "exVZ1KPRO1E0ebmUsKomZpwQL"
consumer_secret = "z45M00EBJOtyyiuivL9K1B7FLFO7ZkS44D6aEiuaxbP4CRh43Q"

t = TweetGrabber(
    myApi = consumer_key,
    sApi = consumer_secret,
    at = access_token,
    sAt = access_token_secret,
    )

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

#Variable to hold whatever twitter user is being classififed
#screen_name = "elonmusk"

def createGML(screen_name):
   # try:
      #  existing_gml = igraph.read(screen_name +'.gml')
     #   print(screen_name + '.gml already exsists')
   # except FileNotFoundError:
    try:
        #print("I got just before the first bit")
        u = api.get_user(screen_name)
        # print("I got to the first bit")
        if u.protected == False:
            print("Scanning activity... of ",screen_name)
            #TESTING USER - collects the users tweets into a CSV file
            t.user_search(user=screen_name, csv_prefix=screen_name)
            
            #read the CSV filed into a Dataframe to put into RetweetParser (Pandas)
            userFrame = pd.read_csv(screen_name + ".csv")
            #TESTING RETWEET PARSER
            r = RetweetParser(userFrame, screen_name)
            #Creates a weighted undirected object (igraph) 
            log_graph = TweetGraph(edge_list= screen_name + ".csv")
            #Lets you see the graph in vector format
            #print(log_graph.e_centrality())
            #Adds thee attribute 'size' to each vertex of the graph - the value of size chanegs depending on the eigencentraility value.
            for key, value in log_graph.e_centrality():
                log_graph.tuple_graph.vs.find(name=key)['size'] = value*20
            # Save the graph in GML format
            print("Building gml...")
            log_graph.tuple_graph.write_gml(f=screen_name+".gml")
            #COMMENT THIS OUT - Plots a viewable graph
            #style = {}
            #style["edge_curved"] = False
            #style["vertex_label"] = log_graph.tuple_graph.vs['name']
            #style["vertex_label_size"] = 13
            #style["bbox"] = (0,0,2000,2000)
            #plot(log_graph.tuple_graph, "testing.png", **style)

            return("successfully created GML for",screen_name)
        else:
            Failiure = ("Private or Unknown")
    except:
        Failiure = ("Failed")
            
#The GML files contain too much irrelivent information which would make training slow
#So we need to find a way to "represent" this information in a more efficient way
#To do this we use Graph2Vec to create denser versions of these graphs
def createConvertedGraph(screen_name): #TODO Add try and except for whether GML exists 
    import networkx as nx
   
    #inserts a line into the GML file showing that this is a multigraph
    igraph_gml = open(screen_name+".gml", 'r') 
    lof = igraph_gml.readlines() 
    igraph_gml.close()
    if lof[4]!="multigraph 1":
        lof.insert(4, "multigraph 1\n")
    igraph_gml = open(screen_name + '.gml', 'w')
    lof="".join(lof)
    igraph_gml.write(lof)
    igraph_gml.close()
    
    try:
        
        
        #Graph2Vec requires the nodes in the graph to be labelled by integers rather than names
        H = nx.read_gml(screen_name + '.gml', label='name')
        convertedgraph = nx.convert_node_labels_to_integers(H)

        #Creates a model to fit the graphs to - 64 columns for graph vectors and a 65th for the label of "bot" or "human"
        embedding_model = Graph2Vec(dimensions = 64)
        
        
    #Fits the graph of the user to the model we just created
        embedding_model.fit([convertedgraph])
    #Stores the result in a Pandas Dataframe
        
        embeddingframe = pd.DataFrame(embedding_model.get_embedding())
        print(embeddingframe)
       
        return embeddingframe
        
        #TemporaryGraphStorageUpdated = TemporaryGraphStorage.append(embeddingframe)
        #TemporaryGraphStorageUpdated.at[0,64] = BotCategory
        #largeData.append(TemporaryGraphStorageUpdated)

    except:
        Failiure = ("Embedding")



#TODO Create pandas dataframe with the usernames and labels of the bot accounts
#iterate through it generating  user graphs for each user
#This will leave us with a folder FULL of user graphs and a CSV file linking to these graphs through their screen_name with a label whether they are a bot or not

#TODO Read CSV file
#read screen_name from CSV file
#read label from CSV file and save as a variable
#Run createConvertedGraph with the screen_name as input
#Add a 65th column to the dataframe "ground truth" and fill that with the label
#Repeat for the whole CSV file

# Takes the CSV file containing the Bot and Human account data (screen name and account type) and converts it into a Pandas Dataframe


#usernames_and_labels = pd.read_csv('twitter_human_bots_dataset.csv', usecols = ["screen_name" , "account_type"]) # TODO: potentially migrate this Pandas dataframe over to MySQL - the 523 error is likely due to a memory limitation implemented by the IDE


#df = pd.DataFrame(usernames_and_labels, columns= ['screen_name','account_type'])

#Uses the databse of bot and human accounts and creates a GML file using their screen_name. 
#This keeps the file linked to the user so we can refer back to our CSV file later for the account type
#This CSV file contains over 37 thousand twitter users. Collecting and converting this data into a readable format is going to take a long time.
#SOLUTION - Run this program multiple times allocating sections of the CSV file to different versions of the program
#This could be done on one computer or several
#x=0



#SQL databse - Screen_Name, Does GML exist?, bot or not?

#usernames_and_labels["GML"]=""
#conn = sqlite3.connect('Processed_Database.db')
#c = conn.cursor()
#c.execute('''CREATE TABLE IF NOT EXISTS GMLCategorized(Screen_Name text PRIMARY KEY, Category INT, GML text)''')


def fill_database():
    for row in df.itertuples():
        screen_name_variable = row.screen_name

        if row.account_type == "bot":
            account_type_boolean = int(1)
        else:
            account_type_boolean = int(0)
        c.execute('''
        INSERT INTO GMLCategorized(Screen_Name, Category)
        VALUES(?, ?)
        ''',
        (screen_name_variable, account_type_boolean)
        )
        conn.commit()


   
#engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
 #                      .format(user = "root",
  #                             pw = "12345",
   #                            db="Processed_Database.db"))


#usernames_and_labels.to_sql('GMlCategorized', con = engine, if_exists = 'append', chunksize = 1000)




#for ind in usernames_and_labels.index + 2000: #Problem occurs at index = 523
 #   print(ind)
  #  new_screen_name = usernames_and_labels['screen_name'][ind]
   # print(ind, new_screen_name, "hello")
    #createGML(new_screen_name)
    #if ind == 2522:
     #   break
    #else:
     #   ()

#count = 0
#from itertools import islice
#for index, row in islice(df.iterows(), 1 , None):
#    count = count + 1
#    new_screen_name = row.screen_name
#    print("hello", new_screen_name, count) # Hits file path limit - run on linux 
#    createGML(new_screen_name)


#count = 0
#for row in df.itertuples():
#    count = count + 1
#    new_screen_name = row.screen_name
#    print("hello", new_screen_name, count) # Hits file path limit - run on linux 
#    createGML(new_screen_name)



#createConvertedGraph("sadGreenRL")

   
#u = api.get_user("bretteldredge")
#print(u.protected)

#CONCERNING USERS
#Scanning activity... of  1051_1051


#classification_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, learning_rate = 0.05, n_estimators = 5000, early_stopping_rounds = 10)
#classification_model.load_model('finalBotDetectorModel.json')




def predict_User(user_name):
    try:
        createGML(user_name)
        user_Frame = createConvertedGraph(user_name)
        classification_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, learning_rate = 0.05, n_estimators = 5000, early_stopping_rounds = 10)
        classification_model.load_model('finalBotDetectorModel.json')
        pred = classification_model.predict(user_Frame) #Objective 4.0
        print(pred)
        user_prediction = str(pred[0]) # Objective 3.0
        return (user_prediction)
    except:
        return ("Failed") # deal with failiure outputs here 
