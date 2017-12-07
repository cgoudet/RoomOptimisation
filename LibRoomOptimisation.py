import numpy as np
import pandas as pd
import pulp
import unittest
from pandas.util.testing import assert_frame_equal

def ArrayFromPulpMatrix2D( x ) : 
    nRows = len(x)
    nCols = len(x[0])
    
    result = np.zeros((nRows, nCols))
    
    for i  in range(nRows):
        for j in range(nCols) :
            result[i][j] = x[i][j].varValue

    return result
#==========
def GetRoomIndex( data) :
    """
    returns the set of columns names which allows identification of a room within the given dataset
    """
    labels = ['etage', 'roomID']
    columns = data.columns
    
    result =[ l for l in labels if l in columns ]
    if not result : raise RuntimeError( 'No tagging possible for a room' )
    return result

#==========
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
def GetCountPerOfficeProp( placements, officeData, persoData, officeTags=['roomID'], persoTags=['inService'] ) :
    """
    Return the number of occurences of persons with a given property (personTag) in offices with another property (officeTag)
    """
   
    persoFilter = pd.pivot_table(persoData, columns=persoTags, index=persoData.index, aggfunc=len).fillna(0).values.T
    officeFilter = pd.pivot_table(officeData, columns=officeTags, index=officeData.index, aggfunc=len).fillna(0).values

    return np.dot( np.dot(persoFilter, placements), officeFilter )

#==========
def GetPRSingleCatMatching( placement, officeData, persoData, propertyName ) :
    """
    Return the weighted agreement between person preferences and office characteristics of categorical variable
    The category the person wants is in the column [option] and the weight in the column [weightOption]
    """

    weightName = 'weight' + propertyName[0].upper() + propertyName[1:]
    
    #returns which property has the office
    officeFilter = pd.pivot_table(officeData.loc[:,[propertyName]], columns=propertyName, index=officeData.index, aggfunc=len).fillna(0)

    #return the weight that a person attribute to being allocaetd category i
    persoFilter = pd.pivot_table(persoData, values=weightName, columns=propertyName, index=persoData.index, aggfunc='sum').fillna(0)

    commonLabels = list(set(persoFilter.columns).intersection(officeFilter.columns))
    officeFilter = officeFilter.loc[:,commonLabels].values
    persoFilter = persoFilter.loc[:,commonLabels].values
    
    #return the properties which have been allocated to each perso
    persoDispo = np.dot( placement, officeFilter )
    
    return np.multiply( persoFilter, persoDispo ).sum(axis=1)
        

#==========
def GetPRCatMatching( placement, officeData, persoData, properties ) :
    """
    Return the weighted agreement between person preferences and office characteristics of categorical variables
    The category the person wants is in the column [option] and the weight in the column [weightOption]
    """
    result = [ GetPRSingleCatMatching( placement, officeData, persoData, opt ) for opt in properties ]      
    return np.array(result).T

#==========
def GetPRBinMatching( placement, officeData, persoData, properties ) :
    """
    Return the weighted agreement between persons preferences and office characteristics
    """
        
    officeProp = officeData.loc[:,properties].values    
    result = np.multiply(persoData.loc[:, properties].values, np.dot(placement, officeProp))
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
def PrintOptimResults( placement, persoData, officeData, happyNeighbours, delta,  spatialProps ) :
    #Print results
    resultFrame = pd.DataFrame({'ID': persoData.index, 'Nom':persoData['Nom']}).set_index('ID')
    resultFrame['office']=-1
    
    x=np.zeros(shape=(len(persoData), len(officeData)))
    hNei = np.zeros(shape=(len(persoData) ,len(happyNeighbours[0])) )
    for iPerso in persoData.index :
        for iRoom in officeData.index : 
            if placement[iPerso][iRoom].varValue   :
                resultFrame.loc[iPerso, 'office'] = iRoom
                x[iPerso][iRoom]=placement[iPerso][iRoom].varValue
                
        for iNei in range(len(hNei[0])) :
            hNei[iPerso][iNei] = happyNeighbours[iPerso][iNei].varValue
    
    print('Total Happyness : ')
    happyness = GetPRBinMatching( x, officeData, persoData, spatialProps).sum(1)
    print('Spatial : ', happyness.sum() )
    resultFrame['happySpat'] = happyness
    
    properties = ['etage']
    happynessFloor =  GetPRCatMatching( x, officeData, persoData, properties ).sum(1)
    print( 'Floor : ', happynessFloor.sum() )
    resultFrame['happyFloor'] = happynessFloor

    print('Neighbours : ', hNei.sum() )
    resultFrame['happyNei'] = hNei.sum(1)
    
    resultFrame['happy'] = resultFrame.loc[:, ['happySpat', 'happyFloor', 'happyNei']].sum(1)
    print(resultFrame)
    
    
    print('Diversité : ', pulp.value(pulp.lpSum(delta) ) )
    
    print('Attributions Bureaux')
    for row in resultFrame.itertuples() :
        print( '%s is given office %i with happyness %2.2f' % (row.Nom,row.office, row.happy))
        
    print( 'total happyness spatial : ', resultFrame['happy'].sum())
    
    
#==========
def NormalizePersoPref( persoData, options ) :
    for opt in options : 
        s = np.sqrt((persoData.loc[:,opt]**2).sum(1))
        persoData.loc[:,opt] = (persoData.loc[:,opt].T / s).T
        
#==========
def GetPeoplePrefMatching( placement, officeData, persoData, properties ):
    result = []
    
    for opt in properties :
     
        usedLabels = persoData[opt].unique()
        print('usedLabels : ', usedLabels)
        weightName = 'weight' + opt[0].upper() + opt[1:]
    
        servPerRoom = GetCountPerOfficeProp( placement, officeData, persoData, officeTags=GetRoomIndex(officeData), persoTags=opt )
    
        officeFilter = pd.pivot_table(officeData, values=officeVal, columns=opt, index=officeData.index, aggfunc='count').fillna(0)
        persoDispo = np.dot( placement, officeFilter.values )


        persoFilter = pd.pivot_table(persoData, values=weightName, columns=opt, index=persoData.index, aggfunc='sum').fillna(0)
        persoFilter = persoFilter.loc[:,officeFilter.columns].values
        
        result.append(np.multiply( persoFilter, persoDispo ).sum(axis=1) )

        result.append(np.multiply( persoFilter, persoDispo ).sum(axis=1) )
    

    return np.array(result).T      

#==========
def TestInput( data, options ) :
    """
    Test wether a dataset has the columns correspondig to options.
    """
    columns = data.columns
    return all( x in columns for x in options)

#==========
def RoomOptimisation( officeData, persoData,
                     diversityTag=[],
                     roomTag=[],
                     spatialBinTag=[],
                     spatialCatTag=[]
                     ) :

    weightVars = spatialCatTag
    persoWeights = [ 'weight'+v[0].upper()+v[1:] for v in weightVars ]
    persoTags = diversityTag + spatialBinTag + persoWeights + spatialCatTag
    isInputOK = TestInput( persoData, persoTags )
    
    officeTag=roomTag + spatialBinTag + spatialCatTag
    isInputOK *= TestInput( officeData, officeTag )
    if not isInputOK : raise RuntimeError('One of these options is not present in the datasets : ', officeTag, persoTags)

    #officeOccupancy_ij = 1 iif person i is seated in office j.
    # Conditions must be imposed on the sum of lines and columns to ensure a unique seat for a person and a unique person on each office.
    officeOccupancy = pulp.LpVariable.matrix("officeOccupancy" ,(list(persoData.index), list(officeData.index)),cat='Binary')
    
    #Create the minimum and maximum happyness as variables
#    minHappy=pulp.LpVariable("minHappy",None,None,cat='Continuous')
#    maxHappy=pulp.LpVariable("maxHappy",None,None,cat='Continuous')
    
   
    
    doDiversity = diversityTag and roomTag
    delta=np.array([])
    if  doDiversity : 
        # Delta counts the number of person from each inService within a room
        Delta = GetCountPerOfficeProp( officeOccupancy, officeData, persoData, officeTags=roomTag, persoTags=diversityTag)
    
        # delta_sr = 1 iif a person from inService s belongs to room r
        # This variable is used to represent the diversity of inServices. The objective function will contain a sum of delta across inServices and rooms.
        # This variable values will be fully constrained by Delta
        nService = len( persoData.groupby(diversityTag) )
        nRooms = len( officeData.groupby( roomTag ) )
        delta = pulp.LpVariable.matrix("delta" ,(np.arange(nService), np.arange(nRooms) ) ,cat='Binary')


#    # legs counts the number of tall people per leftoffice
#    roomTags.append( 'isLeft'  )
#    servTags = [ x for x in ['isTall'] if x in persoProp]
#    legs = None
#    if len(roomTags) and len(servTags) : legs = GetCountPerOfficeProp( officeOccupancy, officeData, persoData, officeTags=roomTags, persoTags=servTags)[1]
#    
    # Define the happyness of one person from its spatial properties. The value of happyness is weight*isAttributedValues
    spatialBinWeights = np.array([])
    if spatialBinTag : spatialBinWeights = GetPRBinMatching( officeOccupancy, officeData, persoData, spatialBinTag )
#
#    # Define the happyness of one person from its neighbours
#    prefNeighbours, roomDistribs = GetNeighbourMatching( officeOccupancy, officeData, persoData )
#    happynessNeighbourShape = (persoData.index.values, range(len(prefNeighbours[0])))
#
#    #Create the amount of happyness from neighbours in a given office for person i
#    happynessNeighbour = pulp.LpVariable.matrix('happynessNeighbour', happynessNeighbourShape , 0, None, cat='Continuous', )
#
#
#    # Create the amount of happyness for floor like variables
    spatialCatWeight = np.array([])

    if spatialCatTag : spatialCatWeight =  GetPRCatMatching( officeOccupancy, officeData, persoData, spatialCatTag )
#     
    #--------------------------------------------
    #Define the optimisation model
    model = pulp.LpProblem("Office setting maximizing happyness", pulp.LpMaximize)


    #Objective function : 
    # maximise the number of service represented in each room
    # Minimize the différence between largest and smallest happyness
    # Maximise the happyness from spatial coordinates
    # maximise the happyness from neigbours
    model += (
            np.sum(delta)
            + spatialBinWeights.sum() 
#            + (minHappy - maxHappy ) 
#            + pulp.lpSum(happynessNeighbour) 
            + np.sum(spatialCatWeight)
            )
    
    #Each perso is set once
    for s  in np.sum(officeOccupancy, axis=0) : model += s <= 1
    
    #Each room is set at least once
    for s in np.sum(officeOccupancy, axis=1) : model += s == 1
        
    if doDiversity : 
        #Constraint of delta
        for s in range( nService ) :
            for j in range( nRooms ) :
                model += delta[s][j] >= Delta[s][j]/len(persoData)
                model += delta[s][j] <= Delta[s][j]

#     #We imose two tall people not to be in front of each other
#    for l in legs : model += l <= 1  
#     
#    for perso in spatialWeights.sum(axis=1) : 
#        model += minHappy <= perso
#        model += maxHappy >= perso
#    
#    for perso in happynessNeighbourShape[0] : 
#        for room in happynessNeighbourShape[1] :
#            model += happynessNeighbour[perso][room]<= prefNeighbours[perso][room]
#            model += happynessNeighbour[perso][room]<= roomDistribs[perso][room]
    
    # Solve the maximisation problem
    model.solve()
    #print('model status : ', pulp.LpStatus[model.status], pulp.value(model.objective) )

 #   PrintOptimResults( officeOccupancy, persoData, officeData, happynessNeighbour, delta, spatialProps )
    
   
    return model, officeOccupancy
#==========
class TestGetCountPerOfficeProp(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'inService':['SI', 'SI', 'RH'], 'isTall':[0,0,0]})
        office = pd.DataFrame({'roomID':[0,0,0], 'isLeft':[0,0,0]})
        
        counts = GetCountPerOfficeProp( np.diag([1,1, 1]), office, perso) 
        self.assertTrue(np.allclose(counts, [[1.],[2.]], rtol=1e-05, atol=1e-08))

#==========
class TestGetPRBinMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'wc':[10,5,2], 'fenetre':[3,8,6]})
        office = pd.DataFrame({'wc':[1,0,1], 'fenetre':[0,1,1]})
        
        agreement = GetPRBinMatching( np.diag([1,1, 1]), office, perso, ['wc','fenetre'])
        self.assertTrue(np.allclose(agreement, [[10, 0],[0, 8], [2, 6]], rtol=1e-05, atol=1e-08))

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
                                   'weightPerso1' : [1,1,1/np.sqrt(37)], 
                                   'weightPerso2' : [0, 0, 6/np.sqrt(37)],
                                   'perso1' : ['Dum1', 'Dum2', 'Dum0'],
                                   'perso2': ['', '', 'Dum1']
                                   })

        assert_frame_equal( perso, perso2 )

#==========
class TestGetPRCatMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'etage':[1,0,2], 'weightEtage':[3,8,6], 'window':[0,0,0] })
        office = pd.DataFrame({'etage':[1,1,2], 'window':[0,0,0]})     
        agreement = GetPRCatMatching( np.diag([1,1, 1]), office, perso, ['etage']) 
        
        self.assertTrue(np.allclose(agreement, [[3],[0],[6]], rtol=1e-05, atol=1e-08))
    
    def test_resultSingle(self ) :
        perso = pd.DataFrame({'etage':[1,0,2], 'weightEtage':[3,8,6], 'window':[0,0,0] })
        office = pd.DataFrame({'etage':[1,1,2], 'window':[0,0,0]})     
        agreement = GetPRSingleCatMatching( np.diag([1,1, 1]), office, perso, 'etage') 

        self.assertTrue(np.allclose(agreement, [3,0,6], rtol=1e-05, atol=1e-08))
          

#==========
class TestGetPeoplePrefMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'inService':['SI','RH', 'SI'], 'weightService':[3,8,6], 'service':['SI', '', 'RH'] })
        office = pd.DataFrame({'roomID':[0,0,0]})     
        #agreement = GetPeoplePrefMatching( np.diag([1,1, 1]), office, perso, ['service']) 

        #self.assertTrue(np.allclose(agreement, [[3],[0],[6]], rtol=1e-05, atol=1e-08))
         
#==========
class TestTestInput( unittest.TestCase):
    def test_result(self) :
        data = pd.DataFrame({'dum0': [1]})
        options = ['dum0', 'dum1']
        
        self.assertFalse( TestInput(data, options)) 
        
        data['dum1']=0
        self.assertTrue( TestInput(data, options))            

#==========
class TestRoomOptimisation( unittest.TestCase ):
    def test_empty(self) : 
        persoData = pd.DataFrame({'inService':['SI','SI','RH']})
        officeData =pd.DataFrame({'roomID':[0,0,0]})
        model, placement = RoomOptimisation( officeData, persoData) 
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), None )
        
        y = ArrayFromPulpMatrix2D( np.array(placement) )
        self.assertTrue(np.allclose(y.sum(1), [1,1,1], rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(y.sum(0), [1,1,1], rtol=1e-05, atol=1e-08))

        
    def test_resultObjectiveDiversity(self) :
        persoData = pd.DataFrame({'inService':['SI','SI','RH']})
        officeData =pd.DataFrame({'roomID':[0,0,0]})
        model, placement = RoomOptimisation( officeData, persoData, roomTag=['roomID'], diversityTag=['inService'] )
    
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 2 )
        
        y = ArrayFromPulpMatrix2D( np.array(placement) )
        self.assertTrue(np.allclose(y.sum(1), [1,1,1], rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(y.sum(0), [1,1,1], rtol=1e-05, atol=1e-08))
        
        
    def test_resultDiversity(self) :
        persoData = pd.DataFrame({'inService':['SI','SI','RH']})
        officeData =pd.DataFrame({'roomID':[0,1,1]})
        model, placement = RoomOptimisation( officeData, persoData, roomTag=['roomID'], diversityTag=['inService'] )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 3 )
        y = ArrayFromPulpMatrix2D( np.array(placement) )
        
        room2 = y[:2,1:].sum(1)
        self.assertEqual(np.linalg.det(np.array([y[:2,0], room2])), 1 )
        
    def test_resultBinSpat(self) : 
        persoData = pd.DataFrame({'window':[1,0,0], 'mur':[0, 0.5, 0]})
        officeData = pd.DataFrame({'window':[1, 0, 0], 'mur':[0, 1, 0]})
        spatialTag = ['mur', 'window' ]
        model, placement = RoomOptimisation( officeData, persoData, spatialBinTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 1.5 )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertTrue(np.allclose(y, np.diag([1,1,1]), rtol=1e-05, atol=1e-08))

    def test_resultCatSpat(self) : 
        persoData = pd.DataFrame({'etage':[1, 1, 1]})
        officeData = pd.DataFrame({'etage':[1, 2, 1]})
        spatialTag = ['etage']
        
        with self.assertRaises(RuntimeError) :
            model, placement = RoomOptimisation( officeData, persoData, spatialCatTag=spatialTag )
        
        persoData['weightEtage'] = [1, 0, 0.5]
        model, placement = RoomOptimisation( officeData, persoData, spatialCatTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 1.5 )
        self.assertEqual( placement[1][1].varValue, 1)
        
    def test_resultBinSpatNegWeight(self) :
        persoData = pd.DataFrame({'window':[1,0], 'mur':[0, -0.5]})
        officeData = pd.DataFrame({'window':[1, 0], 'mur':[0, 1]})
        spatialTag = ['mur', 'window' ]
        model, placement = RoomOptimisation( officeData, persoData, spatialBinTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 0.5 )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertTrue(np.allclose(y, np.diag([1,1]), rtol=1e-05, atol=1e-08))

#==========
if __name__ == '__main__':
    unittest.main()