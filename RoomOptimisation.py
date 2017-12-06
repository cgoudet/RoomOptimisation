import numpy as np
import pandas as pd
import pulp
import unittest
from pandas.util.testing import assert_frame_equal

def GetRoomIndex( data) :
    labels = ['etage', 'roomID']
    columns = data.columns
    
    result =[ l for l in labels if l in columns ]
    if not result : raise RuntimeError( 'No tagging possible for a room' )
    return result

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
    """
    Return the number of occurences of persons with a given property (personTag) in offices with another property (officeTag)
    """
    persoFilter = pd.pivot_table(persoData, values=persoVal, columns=persoTags, index=persoData.index, aggfunc='count').fillna(0).values.T
    officeFilter = pd.pivot_table(officeData, values=officeVal, columns=officeTags, index=officeData.index, aggfunc='count').fillna(0).values
    return np.dot( np.dot(persoFilter, placements), officeFilter )

#==========
def GetNonBinaryMatching( placement, officeData, persoData, properties, officeVal='window' ) :
    keys, vals = zip(*properties.items())

    #Give a signe factor to each property to define its impact on happyness
    persoProp = np.dot( persoData.loc[:, keys], np.diag(vals) )
    
    dicoResult = {}
    
    for opt in keys : 
        weightName = 'weight' + opt[0].upper() + opt[1:]
        persoFilter = pd.pivot_table(persoData, values=weightName, columns=opt, index=persoData.index, aggfunc='sum').fillna(0).values
        persoFilter = persoFilter[:,1:]
        
        officeFilter = pd.pivot_table(officeData, values=officeVal, columns=opt, index=officeData.index, aggfunc='count').fillna(0).values
        persoDispo = np.dot( placement, officeFilter )


     #floorHappyness = np.multiply(floorHappyness.sum(axis=1), persoData['weightEtage'] )   
        print( 'persoDispo : ', persoDispo )
        print( 'placement : ', placement )
        print('persoFilter : ', persoFilter )
        
        dicoResult[opt] = np.multiply( persoFilter, persoDispo ).sum(axis=1)
        
    result = pd.DataFrame( dicoResult )
    print('result : ', result )
    return result

#==========
def GetPropMatching( placement, officeData, persoData, properties ) :
    """
    Return the weighted agreement between persons preferences and office characteristics
    """
    keys, vals = zip(*properties.items())
    
    #Give a signe factor to each property to define its impact on happyness
    persoProp = np.dot( persoData.loc[:, keys], np.diag(vals) )
    
    officeProp = officeData.loc[:,keys].values

    result = np.multiply(persoProp, np.dot(placement, officeProp))
    return result

#==========
def GetPersoPref( persoData ) :
    result = np.zeros((len(persoData), len(persoData)))
    
    prefOrder = [int(c.replace('weightPerso', '')) for c in persoData.columns if 'weightPerso' in c]
    
    dicoName = persoData['Nom'].to_dict()
    dicoName = {v: k for k, v in dicoName.items()}
    
    for rows in persoData.itertuples() :
       for i in prefOrder :
           targetName = getattr(rows,'perso%i'% i)
           targetWeightName = 'weightPerso%i'%i
           if targetName not in dicoName : continue
           indexTarget = dicoName[targetName]
           result[rows.Index][indexTarget] = getattr(rows,targetWeightName)

    return result
    
#==========
def GetNeighbourMatching( placement, officeData, persoData ) :
    """
    return the weighted agreement between a person and its neighbours
    """
    
    prefWeights = GetPersoPref(persoData) 
    
    officeFilter = pd.pivot_table(officeData, values='isLeft', columns=GetRoomIndex(officeData), index=officeData.index, aggfunc='count').fillna(0).values
    persoRoom = np.dot(placement, officeFilter)
    
    result = np.dot( prefWeights, persoRoom )
    
    #result = np.multiply( persoRoom, result)
    return result, persoRoom

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
    
    happyness = GetPropMatching( x, officeData, persoData, spatialProps)
    print('happyness : ', happyness)
    resultFrame['happy'] = happyness.sum(axis=1)
    
    print('Attributions Bureaux')
    for row in resultFrame.itertuples() :
        print( '%s is given office %i with happyness %2.2f' % (row.Nom,row.office, row.happy))
        
    print( 'total happyness : ', resultFrame['happy'].sum())

#==========
def NormalizePersoPref( persoData, options ) :
    for opt in options : 
        s = persoData.loc[:,opt].sum(1)
        persoData.loc[:,opt] = (persoData.loc[:,opt].T / s).T
        
#==========
def main() :
    np.random.seed(12435)

    nPerso=12
    nOffice=20

    # Create randomly generated persons
    options =  ['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'] + ['weightPerso%i'%i for i in range(1,4)] 
    persoProp = { 'service' : [ 'SI', 'RH', 'Achat', 'GRC'], 'isTall' : [0,1, 0] }
    persoData = CreatePersoData( nPerso=nPerso, preferences=options, properties=persoProp)
    NormalizePersoPref( persoData, [['clim', 'mur', 'passage', 'sonnerie', 'wc', 'weightEtage', 'window'], ['weightPerso%i'%i for i in range(1,4)]] )
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
    
    #Create the minimum and maximum happyness as variables
    minHappy=pulp.LpVariable("minHappy",None,None,cat='Continuous')
    maxHappy=pulp.LpVariable("maxHappy",None,None,cat='Continuous')
    
    
    
    # Delta counts the number of person from each service within a room
    roomTags = GetRoomIndex(officeData)
    servTags = [ x for x in ['service'] if x in persoProp]
    Delta = None
    if len(roomTags) and len(servTags) : Delta = GetCountPerOfficeProp( officeOccupancy, officeData, persoData, officeTags=roomTags, persoTags=servTags)

    # delta_sr = 1 iif a person from service s belongs to room r
    # This variable is used to represent the diversity of services. The objective function will contain a sum of delta across services and rooms.
    # This variable values will be fully constrained by Delta
    nService = len(persoProp['service']) if 'service' in persoProp else 1
    nRooms =np.array( [ len(officeProp[sz]) for sz in roomTags ] ).prod()
    delta = pulp.LpVariable.matrix("delta" ,(np.arange(nService), np.arange(nRooms) ) ,cat='Binary')


    # legs counts the number of tall people per leftoffice
    roomTags.append( 'isLeft'  )
    servTags = [ x for x in ['isTall'] if x in persoProp]
    legs = None
    if len(roomTags) and len(servTags) : legs = GetCountPerOfficeProp( officeOccupancy, officeData, persoData, officeTags=roomTags, persoTags=servTags, officeVal='window', persoVal='window')[1]
    
    
    # =============================================================================
    # Define the happyness of one person from its spatial properties
    # Two cases arise :
    #     - If a matching involves a positive feature (window) then the happiness increases by the value attributed to the matching
    #     - If a matching involves a negative feature (sonnerie), then the happiness decreases
    #     - If no matching, wether it was aked or not, the happyness doesn't change
    # =============================================================================
    spatialProps = { 'wc' : -1, 'clim':-1, 'mur':1, 'passage':-1, 'sonnerie':-1, 'window':1 }
    spatialWeights = GetPropMatching( officeOccupancy, officeData, persoData, spatialProps )

    #Define the happyness related to the floor of the office


    
    # Define the happyness of one person from its neighbours
    prefNeighbours, roomDistribs = GetNeighbourMatching( officeOccupancy, officeData, persoData )
    happynessNeighbourShape = (persoData.index.values, range(len(prefNeighbours[0])))

    #Create the amount of happyness from neighbours in a given office for person i
    happynessNeighbour = pulp.LpVariable.matrix('happynessNeighbour', happynessNeighbourShape , 0, None, cat='Continuous', )

    #--------------------------------------------
    #Define the optimisation model
    model = pulp.LpProblem("Office setting maximizing happyness", pulp.LpMaximize)


    #Objective function : 
    # maximise the number of service represented in each room
    # Minimize the différence between largest and smallest happyness
    # Maximise the happyness from spatial coordinates
    # maximise the happyness from neigbours
    model += np.sum(delta) + spatialWeights.sum()    + (minHappy - maxHappy ) + pulp.lpSum(happynessNeighbour)

    
    #Each perso is set once
    for s  in np.sum(officeOccupancy, axis=0) : model += s <= 1
    
    #Each room is set at least once
    for s in np.sum(officeOccupancy, axis=1) : model += s == 1
        
    #Constraint of delta
    for s in range( nService ) :
        for j in range( nRooms ) :
            model += delta[s][j] >= Delta[s][j]/len(persoData)
            model += delta[s][j] <= Delta[s][j]

     #We imose two tall people not to be in front of each other
    for l in legs : model += l <= 1  
     
    for perso in spatialWeights.sum(axis=1) : 
        model += minHappy <= perso
        model += maxHappy >= perso
    
    print( happynessNeighbourShape[0], happynessNeighbourShape[1])
    print( prefNeighbours.shape)
    print( roomDistribs.shape )
    for perso in happynessNeighbourShape[0] : 
        for room in happynessNeighbourShape[1] :
            model += happynessNeighbour[perso][room]<= prefNeighbours[perso][room]
            model += happynessNeighbour[perso][room]<= roomDistribs[perso][room]
    
    # Solve the maximisation problem
    model.solve()
    print('model status : ', pulp.LpStatus[model.status], pulp.value(model.objective) )

    PrintOptimResults( officeOccupancy, persoData, officeData, spatialProps )
    
   
    return 0
#==========
class TestGetCountPerOfficeProp(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'service':['SI', 'SI', 'RH'], 'isTall':[0,0,0]})
        office = pd.DataFrame({'roomID':[0,0,0], 'isLeft':[0,0,0]})
        
        counts = GetCountPerOfficeProp( np.diag([1,1, 1]), office, perso, officeVal='isLeft', persoVal='isTall') 
        self.assertTrue(np.allclose(counts, [[1.],[2.]], rtol=1e-05, atol=1e-08))

#==========
class TestGetPropMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'wc':[10,5,2], 'fenetre':[3,8,6]})
        office = pd.DataFrame({'wc':[1,0,1], 'fenetre':[0,1,1]})
        
        agreement = GetPropMatching( np.diag([1,1, 1]), office, perso, {'wc':-1, 'fenetre':1}) 
        self.assertTrue(np.allclose(agreement, [[-10, 0],[0, 8], [-2, 6]], rtol=1e-05, atol=1e-08))

#==========
class TestGetNeighbourMatching(unittest.TestCase):
    def test_result(self ) :
        perso = pd.DataFrame( {'Nom':['Dum0', 'Dum1', 'Dum2' ],
                               'weightPerso1' : [10,3,1], 
                               'weightPerso2' : [0, 0, 6],
                               'perso1' : ['Dum1', 'Dum2', 'Dum0'],
                               'perso2': ['', '', 'Dum1']
                               })
        
        office = pd.DataFrame( { 'roomID': [1, 0, 0], 'isLeft':[0,0,0] } )
        
        persoHappy, persoRoom = GetNeighbourMatching( np.diag([1,1,1]), office, perso )
        agreement = np.multiply(persoHappy, persoRoom)
        self.assertTrue(np.allclose(agreement, [[0, 0],[3,0], [6, 0]], rtol=1e-05, atol=1e-08))

#==========
class TestGetPersoPref(unittest.TestCase):
    def test_result(self ) :
        perso = pd.DataFrame( {'Nom':['Dum0', 'Dum1', 'Dum2' ],
                               'weightPerso1' : [10,3,1], 
                               'weightPerso2' : [0, 0, 6],
                               'perso1' : ['Dum1', 'Dum2', 'Dum0'],
                               'perso2': ['', '', 'Dum1']
                               })

        pref = GetPersoPref( perso )
        self.assertTrue(np.allclose(pref, [[0, 10, 0],[0, 0, 3], [1, 6, 0]], rtol=1e-05, atol=1e-08))

#==========
class TestNormalizePersoPref(unittest.TestCase):
    def test_result(self ) :
        perso = pd.DataFrame( {'Nom':['Dum0', 'Dum1', 'Dum2' ],
                                   'weightPerso1' : [10,3,1], 
                                   'weightPerso2' : [0, 0, 6],
                                   'perso1' : ['Dum1', 'Dum2', 'Dum0'],
                                   'perso2': ['', '', 'Dum1']
                                   })
        
        NormalizePersoPref(perso, [['weightPerso1', 'weightPerso2']])
        
        perso2 = pd.DataFrame( {'Nom':['Dum0', 'Dum1', 'Dum2' ],
                                   'weightPerso1' : [1,1,1/7], 
                                   'weightPerso2' : [0, 0, 6/7],
                                   'perso1' : ['Dum1', 'Dum2', 'Dum0'],
                                   'perso2': ['', '', 'Dum1']
                                   })

        assert_frame_equal( perso, perso2 )

#==========
class TestGetNonBinaryMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'etage':[1,0,2], 'weightEtage':[3,8,6], 'window':[0,0,0] })
        office = pd.DataFrame({'etage':[1,1,2], 'window':[0,0,0]})
        
        agreement = GetNonBinaryMatching( np.diag([1,1, 1]), office, perso, {'etage':1}) 
        print('agreement : ', agreement)
        self.assertTrue(np.allclose(agreement, [[3],[0],[6]], rtol=1e-05, atol=1e-08))
#==========
if __name__ == '__main__':
    
    
    #main()
    unittest.main()