# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:28:08 2017

@author: Christophe GOUDET
"""

import pandas as pd
import numpy as np
from LibRoomOptimisation import RoomOptimisation, CreateOfficeData, NormalizePersoPref, CreatePersoData

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
    
    selectColumns = ['perso1', 'perso2', 'perso3', 'rawEtage', 'rawWindow', 'rawClim', 'rawSonnerie', 'rawPassage' ]
    data = pd.read_csv( fileName, 
                       header=1,
                       usecols=selectColumns).fillna(naInterpret)
    return data

#==========

#==========
def main():
    
    
    #Read the input data for offices
    officeData = ImportOffices()
    
    persoData = ImportPerso()
    print(persoData)
    return 0

    np.random.seed(12435)

    nPerso=12
    nOffice=20

    # Create randomly generated persons
    options =  ['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'] + ['weightPerso%i'%i for i in range(1,4)] 
    persoProp = { 'inService' : [ 'SI', 'RH', 'Achat', 'GRC'], 'isTall' : [0,1, 0] }
    persoData = CreatePersoData( nPerso=nPerso, preferences=options, properties=persoProp)
    print(persoData)
    NormalizePersoPref( persoData, [['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'], ['weightPerso%i'%i for i in range(1,4)]] )
    print(persoData)
    
     # spatialProps = ['wc', 'clim', 'mur', 'passage', 'sonnerie', 'window' ]
    # Create randomly generated rooms
    officeProp = {'roomID':range(4),
             'isLeft':range(2),
              'wc': [0,0,0,0,1],
              'clim': [0,0,0,0,1],
              'mur': [0,0,0,0,1],
              'passage': [0,0,0,0,1],
              'sonnerie': [0,0,0,0,1],
              'window': [0,0,0,0,1],
              'etage' : [1,2],    
             }
    officeData = CreateOfficeData(nOffice=nOffice,properties=officeProp)
    print(officeData)
 
    RoomOptimisation( officeData, persoData )
    return 0

#==========
if __name__ == '__main__':
    main()