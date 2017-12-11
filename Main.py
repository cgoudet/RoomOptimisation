# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:28:08 2017

@author: Christophe GOUDET
"""

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


import pandas as pd
import numpy as np
from LibRoomOptimisation import RoomOptimisation, PrintOptimResults, ArrayFromPulpMatrix2D
from LibRoomOptimisation import GetPRCatMatching
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
def ImportPerso( fileName='C:\\Users\\Christophe GOUDET\\Google Drive\\Zim\\Projets\\GestionLits\\PersoProperties.csv' ):
    
    naInterpret = {'perso1':'', 
                   'perso2':'', 
                   'perso3':'', 
                   'rawEtage':0,
                   'rawWindow':0,
                   'rawClim':0,
                   'rawSonnerie':0,
                   'rawPassage':0,
                   }
    
    selectColumns = ['inService', 'perso1', 'perso2', 'perso3', 'rawEtage', 'rawWindow', 'rawClim', 'rawSonnerie', 'rawPassage', 'Nom']
                      
    data = pd.read_csv( fileName, 
                       header=1,
                       usecols=selectColumns).fillna(naInterpret)
    return data

#==========
def TransformScale( data, name, newName, increasing=True ) :
    data.loc[data[name]!=0, newName] = data.loc[data[name]!=0, name] * (1.0 if increasing else -1)
    data.fillna({newName:0}, inplace=True)

#==========
def EtageMatching( x ) :
    if x == 0 : return [0, 0]
    elif x < 4 : return [1, 7-2*x]
    else : return [2, (x-3)*2]
#==========
def TransformEtage( data, name ) :
    data['etage'], data['weightEtage'] = zip(*data[name].map(EtageMatching))

#==========
def main():
    
    officeFileName = 'OfficeProperties.csv'
    #Read the input data for offices
    officeData = ImportOffices( )
    #PrintOfficeNumber(officeData)
    
    persoFileName = 'PersoProperties.csv'
    persoData = ImportPerso( )
    
    factors = {'rawWindow':1, 'rawClim':-1, 'rawSonnerie':-1, 'rawPassage':-1 }
    for k, v in factors.items() : TransformScale( persoData, k, k.replace('raw', '').lower(), v==1)
    persoData.loc[:,'Nom'] = persoData['Nom'].apply( lambda x : x.split('.')[0] +'.')
    TransformEtage( persoData, 'rawEtage' )
    for i in range(1, 4) :
        persoData['weightPerso'+str(i)] = 7-2*i
        persoData['inPerso'+str(i)] = persoData['Nom']
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
    
    prBinTag = ['sonnerie', 'window', 'clim', 'passage']
    prCatTag = ['etage']
    ppBinTag = []
    ppCatTag = ['perso1']
    model, placement = RoomOptimisation( officeData, persoData 
                                        , diversityTag=['inService']
                                        , roomTag=['roomID']
                                        , prBinTag=prBinTag
                                        , prCatTag=prCatTag
                                        , ppBinTag=ppBinTag
                                        , ppCatTag=ppCatTag
                                        , printResults=True
                                        )
    
    


    
    return 0

#==========
if __name__ == '__main__':
    main()