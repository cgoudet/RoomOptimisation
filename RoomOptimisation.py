import numpy as np
import pandas as pd
import pulp
import unittest

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

#==========
def CreatePersoData( nPerso=10,
                     properties = {},
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

#==========
def CreateOfficeData( nOffice=12, properties = {} ) : 
    officeData = pd.DataFrame( { 'officeID': np.arange(0, nOffice)})
    for prop, value in properties.items() : officeData[prop] = np.random.choice(value, size=nOffice )
    officeData = officeData.set_index('officeID')
    return officeData

#==========
def GetCountPerOfficeProp( placements, officeData, persoData, officeTags=['roomID'], persoTags=['service'], officeVal='isLeft', persoVal='isTall') :
    persoFilter = pd.pivot_table(persoData, values=persoVal, columns=persoTags, index=persoData.index, aggfunc='count').fillna(0).values.T
    officeFilter = pd.pivot_table(officeData, values=officeVal, columns=officeTags, index=officeData.index, aggfunc='count').fillna(0).values
    return np.dot( np.dot(persoFilter, placements), officeFilter )

#==========
def GetPropMatching( placement, officeData, persoData, properties ) :
    keys, vals = zip(*properties.items())
        
    persoProp = persoData.loc[:, keys]
    persoProp = np.dot( persoProp, np.diag(vals) ).T
    officeProp = officeData.loc[:,keys].values
    return np.dot( np.dot( persoProp, placement), officeProp)

#==========
def PrintOptimResults( placement, persoData, officeData, spatialProps ) :
    #Print results
    resultFrame = pd.DataFrame({'ID': persoData.index, 'Nom':persoData['Nom']}).set_index('ID')
    resultFrame['office']=-1
    
    x=np.zeros(shape=(len(persoData), len(officeData)))
    for iPerso in persoData.index :
        for iRoom in officeData.index : 
            if placement[iPerso][iRoom].varValue   :
                resultFrame.loc[iPerso, 'office'] = iRoom
                x[iPerso][iRoom]=1
    
    #Calcul du happyness par personne
    keys, vals = zip(*spatialProps.items())
    persoProp = persoData.loc[:, keys]
    persoProp = np.dot( persoProp, np.diag(vals) )    
    givenProperties = np.dot(x, officeData.loc[:,keys].values )  
    resultFrame['happy'] = np.multiply(persoProp,givenProperties).sum(axis=1)
    
    print('Attributions Bureaux')
    for row in resultFrame.itertuples() :
        print( '%s is given office %i with happyness %2.2f' % (row.Nom,row.office, row.happy))
        
    print( 'total happyness : ', resultFrame['happy'].sum())
    print(persoData.loc[:, ['service']])
    print(officeData.loc[:,['etage', 'roomID']].sort_values(['etage', 'roomID']))
    print('delta :', GetCountPerOfficeProp( x, officeData, persoData, officeTags=['etage', 'roomID'], persoTags=['service'] ))
    print(pd.pivot_table(officeData, values='isLeft', index=['etage', 'roomID'], aggfunc='count'))
    
    
#==========
def main() :
    np.random.seed(12435)

    nPerso=12
    nOffice=20

    # Create randomly generated persons
    options =  ['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'] + ['weightPerso%i'%i for i in range(1,4)] 
    persoProp = { 'service' : [ 'SI', 'RH', 'Achat', 'GRC'], 'isTall' : [0,1, 0] }
    persoData = CreatePersoData( nPerso=nPerso, preferences=options, properties=persoProp)
    print(persoData)
    
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
    
    
    #officeOccupancy_ij = 1 iif person i is seated in office j.
    # Conditions must be imposed on the sum of lines and columns to ensure a unique seat for a person and a unique person on each office.
    officeOccupancy = pulp.LpVariable.matrix("officeOccupancy" ,(list(persoData.index), list(officeData.index)),cat='Binary')
    
    # delta_sr = 1 iif a person from service s belongs to room r
    # This variable is used to represent the diversity of services. The objective function will contain a sum of delta across services and rooms.
    # This variable values will be fully constrained by Delta
    nService = len(persoProp['service']) if 'service' in persoProp else 1
    nRooms =( len(officeProp['roomID']) if 'roomID' in officeProp else 1 ) * (len(officeProp['etage']) if 'etage' in officeProp else 1)
    delta = pulp.LpVariable.matrix("delta" ,(np.arange(nService), np.arange(nRooms) ) ,cat='Binary')

    # Delta counts the number of person from each service with a room
    roomTags = [ x for x in ['roomID', 'etage'] if x in officeProp]
    servTags = [ x for x in ['service'] if x in persoProp]
    Delta = None
    if len(roomTags) and len(servTags) : Delta = GetCountPerOfficeProp( officeOccupancy, officeData, persoData, officeTags=roomTags, persoTags=servTags)

    # legs counts the number of tall people per leftoffice
    roomTags = [ x for x in ['isLeft', 'roomID', 'etage'] if x in officeProp]
    servTags = [ x for x in ['isTall'] if x in persoProp]
    legs = None
    if len(roomTags) and len(servTags) : legs = GetCountPerOfficeProp( officeOccupancy, officeData, persoData, officeTags=roomTags, persoTags=servTags, officeVal='window', persoVal='window')[1]
    
    
    # =============================================================================
    # Define the happyness of one person from its spatial properties
    # Two cases arise :
    #     - If a matching involves a positive feature (window) then the happiness increases by the value attributed to the matching
    #     - If a matching involves a negative feature (sonnerie), then the happiness decreases
    #     - If no matching, wether it was aked or not, the happyness doesn't change
    #     spatialProps = { 'wc' : -1, 'clim':-1, 'mur':1, 'passage':-1, 'sonnerie':-1, 'window':1, 'etage':1 }
    # =============================================================================
    spatialProps = { 'wc' : -1, 'clim':-1, 'mur':1, 'passage':-1, 'sonnerie':-1, 'window':1, 'etage':1 }
    spatialWeights = GetPropMatching( officeOccupancy, officeData, persoData, spatialProps )

    #--------------------------------------------
    #Define the optimisation model
    model = pulp.LpProblem("Office setting maximizing happyness", pulp.LpMaximize)

    #Objective function : 
    # maximise the number of service represented in each room
    #model += np.sum(delta) + spatialWeights.sum() 
    model +=  np.sum(delta)
    
    #Each perso is set once
    for s  in np.sum(officeOccupancy, axis=0) : model += s <= 1
    
    #Each room is set at least once
    for s in np.sum(officeOccupancy, axis=1) : model += s == 1
        
    #Constraint of delta
    for s in range( nService ) :
        for j in range( nRooms ) :
            model += delta[s][j] >= Delta[s][j]/len(persoData)
            model += delta[s][j] <= Delta[s][j]
        
# =============================================================================
#     #We imose two tall people not to be in front of each other
#     for l in legs : model += l <= 1
#     
# =============================================================================
    
    # Solve the maximisation problem
    model.solve()
    print('model status : ', pulp.LpStatus[model.status], pulp.value(model.objective) )

    PrintOptimResults( officeOccupancy, persoData, officeData, spatialProps )
    
   
    return 0
#==========
class TestGetCountPerOfficeProp(unittest.TestCase):
    
    def test_rresult(self ) :
        perso = pd.DataFrame({'service':['SI', 'SI', 'RH'], 'isTall':[0,0,0]})
        office = pd.DataFrame({'roomID':[0,0,0], 'isLeft':[0,0,0]})
        
        counts = GetCountPerOfficeProp( np.diag([1,1, 1]), office, perso, officeVal='isLeft', persoVal='isTall') 
        self.assertTrue(np.allclose(counts, [[1.],[2.]], rtol=1e-05, atol=1e-08))

#==========
if __name__ == '__main__':
    unittest.main()
    main()
    