# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:28:08 2017

@author: Christophe GOUDET
"""

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import time

import pandas as pd
import numpy as np
from LibRoomOptimisation import RoomOptimisation, PrintOptimResults, ArrayFromPulpMatrix2D
from LibRoomOptimisation import GetPRCatMatching, Constraint
import pulp

def PrintOfficeNumber(officeData):

    img = Image.open("Offices.png")
    draw = ImageDraw.Draw(img)
#    # font = ImageFont.truetype(<font-file>, <font-size>)
#    font = ImageFont.truetype("sans-serif.ttf", 16)
    font = ImageFont.load_default()
#    # draw.text((x, y),"Sample Text",(r,g,b))
    for row in officeData.itertuples() : 
        draw.text((row.xImage, row.yImage), str(row.Index),(0,0,0), size=6)
    img.save('OfficeID.png')

def ImportOffices( fileName='C:\\Users\\Christophe GOUDET\\Google Drive\\Zim\\Projets\\GestionLits\\OfficeProperties.csv' ) :
    dicoOffice = { 'mur':bool,
                  'conf':bool,
                  'window':bool, 
                  'clim':bool,
                  'etage':np.int8,
                  'roomID':int,
                  'isFace':int,
                  }
    data = pd.read_csv( fileName
                       , index_col='ID'
                       ).fillna(0).astype(dicoOffice)
    #data['xImage'], 'yImage']] = np.floor(data[['xImage', 'yImage']])
    return data

#==========
def ImportPerso( fileName='C:\\Users\\Christophe GOUDET\\Google Drive\\Zim\\Projets\\GestionLits\\PersoPref.csv' ):
    
    naInterpret = {
            'rawPerso1':'', 
                   'rawPerso2':'', 
                   'rawPerso3':'', 
                   'rawEtage':0,
                   'rawWindow':0,
                   'rawClim':0,
                   'rawPassage':0,
                   }
    
    selectColumns = ['rawPerso1', 'rawPerso2', 'rawPerso3', 'rawEtage', 'rawWindow', 'rawClim', 'rawPassage', 'Nom']
                      
    data = pd.read_csv( fileName, 
                       header=1,
                       usecols=selectColumns
                       ).fillna(naInterpret)
    return data

#==========
def TransformScale( data, name, newName, increasing=True ) :
    data.loc[data[name]!=0, newName] = data.loc[data[name]!=0, name] * (1.0 if increasing else -1)
    data.fillna({newName:0}, inplace=True)

#==========
def EtageMatching( x ) :
    if x in [0, 3]: return [0, 0]
    elif x < 3 : return [1, 6-3*x]
    else : return [2, (x-3)*3]
#==========
def TransformEtage( data, name ) :
    data['etage'], data['weightEtage'] = zip(*data[name].map(EtageMatching))

#==========
def main():
    
    officeFileName = 'OfficeProperties.csv'
    #Read the input data for offices
    officeData = ImportOffices( officeFileName )
    print(officeData.head())
    #PrintOfficeNumber(officeData)
    
    persoFileName = 'PersoPref.csv'
    persoData = ImportPerso( persoFileName )
    
    factors = {'rawWindow':1, 'rawClim':-1, 'rawPassage':1 }
    for k, v in factors.items() : TransformScale( persoData, k, k.replace('raw', '').lower(), v==1)
    
    persoData.loc[:,'Nom'] = persoData['Nom'].apply( lambda x : x.split('.')[0] +'.')
    TransformEtage( persoData, 'rawEtage' )
    for i in range(1, 4) :
        persoData['weightPerso'+str(i)] = 7-2*i
        persoData['inPerso'+str(i)] = persoData['Nom']
        persoData['perso'+str(i)] = persoData['rawPerso'+str(i)]
     
    persoPropName = 'C:\\Users\\Christophe GOUDET\\Google Drive\\Zim\\Projets\\GestionLits\\PersoProp.csv'
    persoPropName = 'PersoProp.csv'
    persoProp = pd.read_csv( persoPropName )
    print(persoProp.head())
    persoData = pd.merge( persoData, persoProp, on='Nom')
    print(persoData.head())   


#    np.random.seed(12435)
#
#    nPerso=12
#    nOffice=20
#
#    # Create randomly generated persons
#    options =  ['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'] + ['weightPerso%i'%i for i in range(1,4)] 
#    persoProp = { 'inService' : [ 'SI', 'RH', 'Achat', 'GRC'], 'isTall' : [0,1, 0] }
#    persoData = CreatePersoData( nPerso=nPerso, preferences=options, properties=persoProp)
#    print(persoData)
#    NormalizePersoPref( persoData, [['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'], ['weightPerso%i'%i for i in range(1,4)]] )
#    print(persoData)
#    
#     # spatialProps = ['wc', 'clim', 'mur', 'passage', 'sonnerie', 'window' ]
#    # Create randomly generated rooms
#    officeProp = {'roomID':range(4),
#             'isLeft':range(2),
#              'wc': [0,0,0,0,1],
#              'clim': [0,0,0,0,1],
#              'mur': [0,0,0,0,1],
#              'passage': [0,0,0,0,1],
#              'sonnerie': [0,0,0,0,1],
#              'window': [0,0,0,0,1],
#              'etage' : [1,2],    
#             }
#    officeData = CreateOfficeData(nOffice=nOffice,properties=officeProp)
#    print(officeData)
 
    #Repartition without constraint
    #model, placement = RoomOptimisation( officeData, persoData )
    
    #diversity
    #model, placement = RoomOptimisation( officeData, persoData , diversityTag=['inService'], roomTag=['roomID'])
    
    constTag = [Constraint('prBin', 'window', True ),
                Constraint('prBin', 'clim', True ),
                Constraint('prBin', 'passage', True ),
                Constraint('prCat', 'etage', True ),
#                Constraint('ppCat', 'perso1', True, roomTag=['roomID'] ),
#                Constraint('ppCat', 'perso2', True, roomTag=['roomID'] ),
#                Constraint('ppCat', 'perso3', True, roomTag=['roomID'] ),
                ]
    t = time.time()
    model, placement = RoomOptimisation( officeData, persoData 
                                        , diversityTag=['inService']
                                        , constTag=constTag
                                        , printResults=True
                                        )
    
    print('elapsed : ', t -time.time())


    
    return 0

#==========
if __name__ == '__main__':
    main()