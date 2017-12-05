import numpy as np
import pandas as pd
import pulp


def FillRandomPerso( data, persoID, options = [] ) :
    """ 
    
    Change the data according to random options
    
    Arguments:
        data {DataFrame} -- Data to modify
        persoID {int} -- Index of the person to change properties
    """
    
    nFilled = np.random.randint(1, len(options))   
    choices = np.random.choice(options, size=nFilled, replace=False )

    for i, v in enumerate(choices) :
        data.loc[persoID,v] = len(options)-i
        
        if 'weightPerso' in v :
            nPerso = int(v.replace('weightPerso', ''))
            nameCol = 'perso%d'%nPerso
            friendID = np.random.randint(0, len(data))
            data.loc[persoID, nameCol] = data.loc[friendID,'Nom']
            
        if v=='weightEtage' : data.loc[persoID,'etage'] = np.random.randint(1, 3)

def CreatePersoData( nPerso=10,
                     properties = { 'services' : [ 'SI', 'RH', 'Achat', 'GRC'], 'isTall' : [0,1, 0] },
                     preferences = ['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'] 
                     ) : 
    """"
    
    Generate a dataframe simulating a distribution of perso characteristics
    
    Keyword Arguments:
        nPerso {int} -- Number of characters to generate (default: {10})
    """
    
    dicoPersos = {'Nom' : ['Dummy%d'%(i) for i in range(nPerso)]}

    persoData = pd.DataFrame( dicoPersos)

    for prop, value in properties.items() : persoData[prop] = np.random.choice(value, size=nPerso)

    for col in preferences + ['etage']: persoData[col]=0

    persoCol = [ 'perso%i'%int(pref.replace('weightPerso', '')) for pref in preferences if 'weightPerso' in pref]
    for col in persoCol : persoData[col]=''

    #simulate the preferences
    for iPerso in range(nPerso) : FillRandomPerso( persoData, iPerso, preferences)

    return persoData


def CreateOfficeData( nOffice=12, properties = {} ) : 
    officeData = pd.DataFrame( { 'officeID': np.arange(0, nOffice)})
    for prop, value in properties.items() : officeData[prop] = np.random.choice(value, size=nOffice )
    officeData = officeData.set_index('officeID')
    return officeData

def main() :
    np.random.seed(12435)

    options =  ['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'] + ['weightPerso%i'%i for i in range(1,4)] 

    persoData = CreatePersoData( nPerso=10, preferences=options)
    print(persoData)
    
    return 0

#==========
if __name__ == '__main__':
    main()