import numpy as np
import pandas as pd
import pulp
import unittest
from pandas.util.testing import assert_frame_equal
from PIL import Image, ImageFont, ImageDraw


def SetPRConstraint( model, weights, bound, up=True ) : 
    s = weights.shape
    for i in range(s[0]) :
        for j in range(s[1]) : 
            if not weights[i][j] : continue
            if up : model += weights[i][j] <= bound
            else : model += weights[i][j] >= bound

#==========
def SetPRBinConstraint( model, placement, officeData, persoData, tags, bound, up=True ) :
    weights = GetPRBinMatching( placement, officeData, persoData, tags )
    SetPRConstraint(model, weights, bound, up )
    
#==========
def SetPRCatConstraint( model, placement, officeData, persoData, tags, bound, up=True ) :
    weights =  GetPRCatMatching( placement, officeData, persoData, tags )
    for w in weights :  SetPRConstraint( model, w, bound, up)
        
#==========
class TestSetConstraint(unittest.TestCase):
    def setUp( self ) : 
        self.persoData = pd.DataFrame({'inPhone':[0, 1, 1], 
                                       'weightPhone':[-2, 0, 1],
                                       'window' : [1, 0, 0],
                                       'clim' : [0, 0, 1],
                                       'etage' : [1, 0, 2],
                                       'weightEtage': [1, 1, 0]
                                        })
        self.officeData = pd.DataFrame({'roomID':[1,0,0],
                                        'window':[1, 0, 0],
                                        'clim':[0, 0, 1],
                                        'etage':[1, 0, 0],
                                        })    
        
        self.pulpVars = pulp.LpVariable.matrix("officeOccupancy" ,(np.arange(3), np.arange(3)),cat='Binary')
        self.model = pulp.LpProblem("Office setting maximizing happyness", pulp.LpMaximize)

        #SetConstraints
        for s  in np.sum(self.pulpVars, axis=0) : self.model += s <= 1
        for s in np.sum(self.pulpVars, axis=1) : self.model += s == 1    
    
    def test_PRCatConstraintUp(self):
        properties = ['etage']
        bound = 0
        weights = GetPRCatMatching( self.pulpVars, self.officeData, self.persoData, properties)
        SetPRConstraint( self.model, weights, bound, up=True)
        self.model.solve()        
        x = ArrayFromPulpMatrix2D(self.pulpVars)
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertEqual(x[1][0], 1)

    def test_PRCatConstraintUpInf(self):
        properties = ['etage']
        bound = 0
        self.persoData.loc[0,'etage'] = 0
        weights = GetPRCatMatching( self.pulpVars, self.officeData, self.persoData, properties)
        SetPRConstraint( self.model, weights, bound, up=True)
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Infeasible' )
        
    def test_PRCatConstraintLow(self):
        properties = ['etage']
        bound=1
        weights = GetPRCatMatching( self.pulpVars, self.officeData, self.persoData, properties)
        SetPRConstraint(self.model, weights, bound, up=False)
        self.model.solve()
        x = ArrayFromPulpMatrix2D(self.pulpVars)
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertEqual(x[0][0], 1)

    def test_PPConstraint(self) :
        tag='phone'
        
        (wish, dispo) = GetPPBinMatching(self.pulpVars, self.officeData, self.persoData, [tag] )
        s = wish.shape

        pulpMaxVars =  pulp.LpVariable.matrix( 'ppBin', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        K = np.fabs(self.persoData[['weight' + tag[0].upper() + tag[1:]]]).values.sum()
        pulpBinVars =  pulp.LpVariable.matrix( tag+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        
        self.model += pulp.lpSum(pulpMaxVars)
        
        SetPPConstraint( self.model, wish, dispo, pulpMaxVars, pulpBinVars, K ) 
        
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertEqual(pulp.value(self.model.objective), 1 )

    def test_PPConstraintUpBound(self) :
        tag='phone'
        
        (wish, dispo) = GetPPBinMatching(self.pulpVars, self.officeData, self.persoData, [tag] )
        s = wish.shape

        pulpMaxVars =  pulp.LpVariable.matrix( 'ppBin', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        K = np.fabs(self.persoData[['weight' + tag[0].upper() + tag[1:]]]).values.sum()
        pulpBinVars =  pulp.LpVariable.matrix( tag+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        
        self.model += pulp.lpSum(pulpMaxVars)
        
        SetPPConstraint( self.model, wish, dispo, pulpMaxVars, pulpBinVars, K ) 
        
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        self.assertEqual(pulp.value(self.model.objective), 1 )

        #Test not yet prepared
        self.assertTrue(False)
 
    def test_PPConstraintBoundInf(self) :
        tag='phone'
        
        (wish, dispo) = GetPPBinMatching(self.pulpVars, self.officeData, self.persoData, [tag] )
        s = wish.shape

        pulpMaxVars =  pulp.LpVariable.matrix( 'ppBin', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        K = np.fabs(self.persoData[['weight' + tag[0].upper() + tag[1:]]]).values.sum()
        pulpBinVars =  pulp.LpVariable.matrix( tag+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        
        self.model += pulp.lpSum(pulpMaxVars)
        
        SetPPConstraint( self.model, wish, dispo, pulpMaxVars, pulpBinVars, K, -1, -100 ) 
        
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Infeasible' )

        
    #===========    
    def test_PRBinLow(self ) :
    
        tags=['window', 'clim']
        bound = 1
        
        SetPRBinConstraint( self.model, self.pulpVars, self.officeData, self.persoData, tags, bound, up=False )
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        x = ArrayFromPulpMatrix2D(self.pulpVars)
        self.assertTrue(np.allclose( np.diag([1,1,1]), x, rtol=1e-05, atol=1e-08))
    
    def test_PRBinLowInf(self) : 
        self.persoData.loc[1,'clim']=1
        tags=['window', 'clim']
        bound = 1
        
        SetPRBinConstraint( self.model, self.pulpVars, self.officeData, self.persoData, tags, bound, up=False )
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Infeasible' )

    def test_PRBinUp(self ) :
    
        tags=['window', 'clim']
        bound = 0       
        SetPRBinConstraint( self.model, self.pulpVars, self.officeData, self.persoData, tags, bound, up=True )       
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Optimal' )
        x = ArrayFromPulpMatrix2D(self.pulpVars)     
        self.assertEqual( 0, x[0,0])
        self.assertEqual( 0, x[2,2])
 
    def test_PRBinUpInf(self ) :

        self.persoData.loc[:,'clim']=1
        tags=['window', 'clim']
        bound = 0
        SetPRBinConstraint( self.model, self.pulpVars, self.officeData, self.persoData, tags, bound, up=True )       
        self.model.solve()        
        self.assertEqual(pulp.LpStatus[self.model.status], 'Infeasible' )
 
#===========
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
    persoFilter = pd.pivot_table(persoData.loc[:,persoTags], columns=persoTags, index=persoData.index, aggfunc=len).fillna(0).values.T
    officeFilter = pd.pivot_table(officeData.loc[:,officeTags], columns=officeTags, index=officeData.index, aggfunc=len).fillna(0).values

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
def GetPPBinSingleMatching( placement, officeData, persoData, option, roomID=[] ) :
    """
    Return the wishing and dispo arrays for a single matching between binary property
    """
    
    if not roomID : roomID = GetRoomIndex(officeData)

    weightName = 'weight' + option[0].upper() + option[1:]
    inName = weightName.replace('weight', 'in' )
    
    officeFilter = pd.pivot_table(officeData.loc[:,roomID], columns=roomID, index=officeData.index, aggfunc=len).fillna(0)
    officeFilter = np.dot(placement, officeFilter)
    
    weightPerRoom = np.dot( persoData[weightName].values.T, officeFilter )
    
    persoDispo = persoData[inName]
    persoDispo = np.dot( persoDispo.T, officeFilter )
    
    return (weightPerRoom, persoDispo)


        
#==========
def GetPPBinMatching( placement, officeData, persoData, properties, roomID=[] ) :
    totWish=[]
    totDispo=[]
    
    for opt in properties : 
        wish, dispo = GetPPBinSingleMatching(placement, officeData, persoData, opt, roomID)
        totWish.append(wish)
        totDispo.append(dispo)

    return (np.array(totWish), np.array(totDispo))


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
def PrintOptimResults( placement
                      , persoData
                      , officeData
                      , diversityTag=[]
                      , roomTag=[]
                      , prBinTag=[]
                      , prCatTag=[]
                      , ppBinTag=[]
                      , ppCatTag=[]
                      ) : #, happyNeighbours, delta,  spatialProps ) :
    #Print results
    resultFrame = pd.DataFrame({'ID': persoData.index, 'Nom':persoData['Nom']}).set_index('ID')
    resultFrame['office']=-1
    
    x=ArrayFromPulpMatrix2D(placement)
#    hNei = np.zeros(shape=(len(persoData) ,len(happyNeighbours[0])) )
    for i, iPerso in enumerate(persoData.index) :
        for j, iRoom in enumerate(officeData.index) : 
            if x[i][j] : resultFrame.loc[iPerso, 'office'] = iRoom
    
    print('diversityTag : ', diversityTag)
    if diversityTag : 
        Delta = GetCountPerOfficeProp( x, officeData, persoData, officeTags=roomTag, persoTags=diversityTag)
        Delta = np.min(Delta, 1)
        print('diversityTags : ' + ' '.join(diversityTag)+'\n', Delta)
        print('diversityObjective : ', Delta.sum())
        
    
    if ppBinTag : 
        prBinWeights = pd.DataFrame( GetPRBinMatching( x, officeData, persoData, prBinTag ), columns=prBinTag, index=persoData.index )
        print( 'prBinWeights : '+' '.join(prCatTag) + '\n', prBinWeights)
        prBinObjective = prBinWeights.values.sum()
        print( 'prBinObjective : ', prBinObjective )
        
    if ppCatTag : 
        prCatWeights = pd.DataFrame( GetPRCatMatching( x, officeData, persoData, prCatTag ), columns = prCatTag, index=persoData.index )
        print( 'prCatWeights : ' + ' '.join(prCatTag) + '\n', prCatWeights)
        prCatObjective = prCatWeights.values.sum()
        print( 'prCatObjective : ', prCatObjective )
    
    for opt in ppCatTag : 
        (wish, dispo) = GetPPCatSingleMatching( x, officeData, persoData, opt, roomID=roomTag )
        m = np.multiply(wish, dispo)
        print( 'ppCatWeights : ' + opt +'\n', m )
        print( 'ppCatObjective : ' + opt + ' : ', m.sum())
    
#        for iNei in range(len(hNei[0])) :
#            hNei[iPerso][iNei] = happyNeighbours[iPerso][iNei].varValue
    
#    print('Total Happyness : ')
#    happyness = GetPRBinMatching( x, officeData, persoData, spatialProps).sum(1)
#    print('Spatial : ', happyness.sum() )
#    resultFrame['happySpat'] = happyness
#    
#    properties = ['etage']
#    happynessFloor =  GetPRCatMatching( x, officeData, persoData, properties ).sum(1)
#    print( 'Floor : ', happynessFloor.sum() )
#    resultFrame['happyFloor'] = happynessFloor
#
#    print('Neighbours : ', hNei.sum() )
#    resultFrame['happyNei'] = hNei.sum(1)
#    
#    resultFrame['happy'] = resultFrame.loc[:, ['happySpat', 'happyFloor', 'happyNei']].sum(1)
#    print(resultFrame)
#    
#    
#    print('Diversité : ', pulp.value(pulp.lpSum(delta) ) )
    
    print('Attributions Bureaux')
    for row in resultFrame.itertuples() :
        print( '%s is given office %i with happyness %2.2f' % (row.Nom,row.office, 0))
    
    #resultFrame[['xImage', 'yImage']] = officeData[['xImage', 'yImage']]
    #DrawOutput( resultFrame, officeData[['xImage', 'yImage']] )
   # print( 'total happyness spatial : ', resultFrame['happy'].sum())

#==========
def DrawOutput( repartition, officeCoord ) :

    img = Image.open("Offices.png")
    draw = ImageDraw.Draw(img)
#    # font = ImageFont.truetype(<font-file>, <font-size>)
#    font = ImageFont.truetype("sans-serif.ttf", 16)
    font = ImageFont.load_default()
#    # draw.text((x, y),"Sample Text",(r,g,b))
    for row in repartition.itertuples() : 
        x = officeCoord.loc[row.office,'xImage']
        y = officeCoord.loc[row.office,'yImage']
        draw.text((x, y), str(row.Nom),(0,0,0),font=font)
    img.save('sample-out.png')
    
#==========
def NormalizePersoPref( persoData, options ) :
    for opt in options : 
        s = np.sqrt((persoData.loc[:,opt]**2).sum(1))
        persoData.loc[:,opt] = (persoData.loc[:,opt].T / s).T
        
#==========
def GetPPCatSingleMatching( placement, officeData, persoData, option, roomID=[] ) :
        if not roomID : roomID = GetRoomIndex(officeData)
        inOption = 'in' + option[0].upper() + option[1:]
        weightName = 'weight' + option[0].upper() + option[1:]
        commonLabels = sorted(list(set(persoData[option]).intersection(persoData[inOption])))
        
        #self property of each person
        persoInOption = pd.pivot_table( persoData.loc[:, [inOption]], columns=[inOption], index=persoData.index, aggfunc=len).fillna(0)
        persoInOption = persoInOption[commonLabels].values      
        #weight for preferences of each person
        persoFilter = pd.pivot_table( persoData.loc[:,[weightName,option]], values=weightName, columns=[option], index=persoData.index, aggfunc='sum').fillna(0)
        persoFilter = persoFilter.loc[:,commonLabels]
        
        
        officeFilter = pd.pivot_table(officeData.loc[:, roomID], columns=roomID, index=officeData.index, aggfunc=len).fillna(0)
        officeFilter = np.dot( placement, officeFilter.values )
        persoDispo = np.dot( persoInOption.T, officeFilter)
        persoWish = np.dot( persoFilter.values.T, officeFilter)     

        return (persoWish, persoDispo)


        
#==========
def MultiplyWithFilter( weights, filt ) : 
    filt[filt!=0] = 1
    return np.multiply(weights, filt)


#==========
def GetPPCatMatching( placement, officeData, persoData, properties, roomID=[] ):
    result = [GetPPCatSingleMatching( placement, officeData, persoData, opt, roomID ) for opt in properties ]
    return result   

#==========
def TestInput( data, options ) :
    """
    Test wether a dataset has the columns correspondig to options.
    """
    columns = data.columns
    return all( x in columns for x in options)

#==========
def SetPPConstraint( model, wish, dispo, pulpMaxVars, pulpBinVars, K, bound = 0, value=0) :
    s = wish.shape
    for i in range(s[0]) :
        for j in range(s[1]) :
            model += pulpMaxVars[i][j] <= 2 * K + wish[i][j] - 2*K * pulpBinVars[i][j]
            model += pulpBinVars[i][j] <= dispo[i][j]
            model += pulpBinVars[i][j] >= dispo[i][j]/K
            model += pulpMaxVars[i][j] <= K * dispo[i][j]
            model += pulpMaxVars[i][j] >= - K * dispo[i][j]
            model += pulpMaxVars[i][j] >= -2 * K + wish[i][j] + 2*K * pulpBinVars[i][j]
            if bound>0 : model += pulpMaxVars[i][j] >= value
            elif bound<0 : model += pulpMaxVars[i][j] <= value

#==========
def RoomOptimisation( officeData, persoData,
                     diversityTag=[],
                     roomTag=[],
                     prBinTag=[],
                     prCatTag=[],
                     consSpatBinTag=[],
                     ppCatTag=[],
                     ppBinTag=[],
                     minimize=True,
                     printResults=False,
                     ) :

    weightVars = prCatTag
    persoWeights = [ 'weight'+v[0].upper()+v[1:] for v in weightVars ]
    persoTags = diversityTag + prBinTag + persoWeights + prCatTag
    isInputOK = TestInput( persoData, persoTags )
    
    officeTag=roomTag + prBinTag + prCatTag
    isInputOK *= TestInput( officeData, officeTag )
    if not isInputOK : raise RuntimeError('One of these options is not present in the datasets : ', officeTag, persoTags)

    #officeOccupancy_ij = 1 iif person i is seated in office j.
    # Conditions must be imposed on the sum of lines and columns to ensure a unique seat for a person and a unique person on each office.
    officeOccupancy = pulp.LpVariable.matrix("officeOccupancy" ,(list(persoData.index), list(officeData.index)),cat='Binary')
 
    
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

 
    # Define the happyness of one person from its spatial properties. The value of happyness is weight*isAttributedValues
    spatialBinWeights = np.array([])
    if prBinTag : spatialBinWeights = GetPRBinMatching( officeOccupancy, officeData, persoData, prBinTag )

    ppCatVars = []    
    for opt in ppCatTag : 
        (wish, dispo) = GetPPCatSingleMatching( officeOccupancy, officeData, persoData, opt, roomID=roomTag )
        s = wish.shape
        pulpVars =  pulp.LpVariable.matrix( opt+'Max', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        pulpBinVars =  pulp.LpVariable.matrix( opt+'Bin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        K = np.fabs(persoData['weight'+opt[0].upper()+opt[1:]]).sum()
        ppCatVars.append( (pulpVars, pulpBinVars, wish, dispo, K))
        
    pulp.lpSum(None)
    ppBinPulpVars = None
    ppBinPulpBinVars = None
    ppBinWish = np.array([])
    ppBinDispo = np.array([])
    ppBinK=None
    if len(ppBinTag) : 
        (ppBinWish, ppBinDispo) = GetPPBinMatching( officeOccupancy, officeData, persoData, ppBinTag, roomID=roomTag )
        s = ppBinWish.shape
        ppBinPulpVars =  pulp.LpVariable.matrix( 'ppBinMax', (np.arange(s[0]), np.arange(s[1])), cat='Continuous' )
        ppBinPulpBinVars =  pulp.LpVariable.matrix( 'ppBinBin', (np.arange(s[0]), np.arange(s[1])), cat='Binary' )
        ppBinK = np.fabs(persoData[['weight' + v[0].upper() + v[1:] for v in ppBinTag]]).values.sum()   


#    # Define the happyness of one person from its neighbours
#    prefNeighbours, roomDistribs = GetNeighbourMatching( officeOccupancy, officeData, persoData )
#    happynessNeighbourShape = (persoData.index.values, range(len(prefNeighbours[0])))
#
#    #Create the amount of happyness from neighbours in a given office for person i
#    happynessNeighbour = pulp.LpVariable.matrix('happynessNeighbour', happynessNeighbourShape , 0, None, cat='Continuous', )
#

#    # Create the amount of happyness for floor like variables
    spatialCatWeight = np.array([])
    if prCatTag : spatialCatWeight =  GetPRCatMatching( officeOccupancy, officeData, persoData, prCatTag )
#     
    #--------------------------------------------
    #Define the optimisation model
    model = pulp.LpProblem("Office setting maximizing happyness", pulp.LpMaximize)


    #Objective function : 
    model += (
            np.sum(delta)
            + spatialBinWeights.sum() 
#            + pulp.lpSum(happynessNeighbour) 
            + np.sum(spatialCatWeight)
            + pulp.lpSum(v[0]for v in ppCatVars)
            + pulp.lpSum(ppBinPulpVars)
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


    for (pulpVars, pulpBinVars, wish, dispo, K ) in ppCatVars :
            SetPPConstraint( model, wish, dispo, pulpVars, pulpBinVars, K )
    
    SetPPConstraint( model, ppBinWish, ppBinDispo, ppBinPulpVars, ppBinPulpBinVars, ppBinK )

           
#    # legs counts the number of tall people per leftoffice
#    roomTags.append( 'isLeft'  )
#    servTags = [ x for x in ['isTall'] if x in persoProp]
#    legs = None
#    if len(roomTags) and len(servTags) : legs = GetCountPerOfficeProp( officeOccupancy, officeData, persoData, officeTags=roomTags, persoTags=servTags)[1]

#     #We imose two tall people not to be in front of each other
#    for l in legs : model += l <= 1  
#     

#    
#    for perso in happynessNeighbourShape[0] : 
#        for room in happynessNeighbourShape[1] :
#            model += happynessNeighbour[perso][room]<= prefNeighbours[perso][room]
#            model += happynessNeighbour[perso][room]<= roomDistribs[perso][room]
    
    # Solve the maximisation problem
    if minimize : model.solve()

    if printResults : 
        PrintOptimResults( officeOccupancy
                      , persoData
                      , officeData
                      , diversityTag
                      , roomTag
                      , prBinTag
                      , prCatTag
                      , ppBinTag
                      , ppCatTag
                      )
        print('\n==========\n')
        print('model status : ', pulp.LpStatus[model.status] )
        print('objective : ', pulp.value(model.objective) )

   
    return model, officeOccupancy

#==========
class TestMultiplyWithFilter( unittest.TestCase ) :
    def test_result(self):
        x = np.array( [1, 2, 3] )
        f = np.array( [1, 0, 4] )
        y = MultiplyWithFilter(x, f )
        self.assertTrue(np.allclose( [1,0,3], y, rtol=1e-05, atol=1e-08))
        
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
        perso = pd.DataFrame({'etage':[1,0,2], 'weightEtage':[3,8,6] })
        office = pd.DataFrame({'etage':[1,1,2]})     
        agreement = GetPRCatMatching( np.diag([1,1, 1]), office, perso, ['etage']) 
        
        self.assertTrue(np.allclose(agreement, [[3],[0],[6]], rtol=1e-05, atol=1e-08))
    
    def test_resultSingle(self ) :
        perso = pd.DataFrame({'etage':[1,0,2], 'weightEtage':[3,8,6], 'window':[0,0,0] })
        office = pd.DataFrame({'etage':[1,1,2], 'window':[0,0,0]})     
        agreement = GetPRSingleCatMatching( np.diag([1,1, 1]), office, perso, 'etage') 

        self.assertTrue(np.allclose(agreement, [3,0,6], rtol=1e-05, atol=1e-08))
          
    def test_resultNegWeight(self ) :
        perso = pd.DataFrame({'etage':[1,0,2], 'weightEtage':[-3,8,-6] })
        office = pd.DataFrame({'etage':[1,1,2]})     
        agreement = GetPRCatMatching( np.diag([1,1, 1]), office, perso, ['etage']) 
        
        self.assertTrue(np.allclose(agreement, [[-3],[0],[-6]], rtol=1e-05, atol=1e-08))


class TestTestInput( unittest.TestCase):
    def test_result(self) :
        data = pd.DataFrame({'dum0': [1]})
        options = ['dum0', 'dum1']
        
        self.assertFalse( TestInput(data, options)) 
        
        data['dum1']=0
        self.assertTrue( TestInput(data, options))            

#==========
class TestGetPPCatSingleMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'inService':['SI','RH', 'SI'], 'weightService':[3,8,6], 'service':['SI', '', 'RH'] })
        office = pd.DataFrame({'roomID':[0,0,0]})     
        (persoWish, persoDispo) = GetPPCatSingleMatching( np.diag([1,1, 1]), office, perso, 'service')
        self.assertTrue(np.allclose( [[6], [3]], persoWish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1], [2]], persoDispo, rtol=1e-05, atol=1e-08))
    
    def test_resultSelfLiking(self ) :
        #Self liking is accepted
        perso = pd.DataFrame({'inService':['SI','RH'], 'weightService':[3,8], 'service':['RH', 'RH'] })
        office = pd.DataFrame({'roomID':[0,1]})     
        (persoWish, persoDispo) = GetPPCatSingleMatching( np.diag([1,1]), office, perso, 'service')
        self.assertTrue(np.allclose( [[3, 8]], persoWish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[0, 1]], persoDispo, rtol=1e-05, atol=1e-08))
        
    def test_resultDiffNumberRoomPerso(self ) :
        perso = pd.DataFrame({'inService':['SI','RH', 'SI'], 'weightService':[3,8,6], 'service':['SI', '', 'RH'] })
        office = pd.DataFrame({'roomID':[0,0,0,0]})     
        (persoWish, persoDispo) = GetPPCatSingleMatching( np.array([[1,0,0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), office, perso, 'service')
        self.assertTrue(np.allclose( [[6], [3]], persoWish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1], [2]], persoDispo, rtol=1e-05, atol=1e-08))


#==========
class TestGetPPCatMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'inService':['SI','RH', 'SI'], 'weightService':[3,8,6], 'service':['SI', '', 'RH'] })
        office = pd.DataFrame({'roomID':[0,0,0]})     
        (wish, dispo) = GetPPCatMatching( np.diag([1,1, 1]), office, perso, ['service'])[0]
        self.assertTrue(np.allclose( [[6], [3]], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1], [2]], dispo, rtol=1e-05, atol=1e-08))


#==========
class TestGetPPBinSingleMatching(unittest.TestCase):
    
    def test_resultNegWeight(self ) :
        perso = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[-2, 0, 0] })
        office = pd.DataFrame({'roomID':[0,0,1]})     
        (wish, dispo) = GetPPBinSingleMatching( np.diag([1,1, 1]), office, perso, 'phone' )
        self.assertTrue(np.allclose( [-2, 0], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [1, 1], dispo, rtol=1e-05, atol=1e-08))
        
    def test_result(self ) :
        perso = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[2, 0, 0] })
        office = pd.DataFrame({'roomID':[0,0,1]})     
        (wish, dispo) = GetPPBinSingleMatching( np.diag([1,1, 1]), office, perso, 'phone' )
        self.assertTrue(np.allclose( [2, 0], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [1, 1], dispo, rtol=1e-05, atol=1e-08))

#==========
class TestGetPPBinMatching(unittest.TestCase):
    
    def test_result(self ) :
        perso = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[-2, 0, 0], 'inSmoke':[1, 0, 1], 'weightSmoke':[0, -5, 0] })
        office = pd.DataFrame({'roomID':[0,0,1]})     
        (wish, dispo) = GetPPBinMatching( np.diag([1,1, 1]), office, perso, ['phone', 'smoke' ])
        self.assertTrue(np.allclose( [[-2, 0],[-5, 0]], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1, 1],[1, 1]], dispo, rtol=1e-05, atol=1e-08))

    def test_resultSelfLiking(self ) :
        perso = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[-2, 1, 0], 'inSmoke':[1, 0, 1], 'weightSmoke':[1, -5, 0] })
        office = pd.DataFrame({'roomID':[0,0,1]})     
        (wish, dispo) = GetPPBinMatching( np.diag([1,1, 1]), office, perso, ['phone', 'smoke' ])
        self.assertTrue(np.allclose( [[-1, 0],[-4, 0]], wish, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose( [[1, 1],[1, 1]], dispo, rtol=1e-05, atol=1e-08))

#========== 

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
        model, placement = RoomOptimisation( officeData, persoData, prBinTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 1.5 )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertTrue(np.allclose(y, np.diag([1,1,1]), rtol=1e-05, atol=1e-08))

    def test_resultCatSpat(self) : 
        persoData = pd.DataFrame({'etage':[1, 1, 1]})
        officeData = pd.DataFrame({'etage':[1, 2, 1]})
        spatialTag = ['etage']
        
        with self.assertRaises(RuntimeError) :
            model, placement = RoomOptimisation( officeData, persoData, prCatTag=spatialTag )
        
        persoData['weightEtage'] = [1, 0, 0.5]
        model, placement = RoomOptimisation( officeData, persoData, prCatTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 1.5 )
        self.assertEqual( placement[1][1].varValue, 1)
        
    def test_resultBinSpatNegWeight(self) :
        persoData = pd.DataFrame({'window':[1,0], 'mur':[0, -0.5]})
        officeData = pd.DataFrame({'window':[1, 0], 'mur':[0, 1]})
        spatialTag = ['mur', 'window' ]
        model, placement = RoomOptimisation( officeData, persoData, prBinTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 0.5 )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertTrue(np.allclose(y, np.diag([1,1]), rtol=1e-05, atol=1e-08))
    
    def test_resultPPCatMatching(self) :
        persoData = pd.DataFrame({'inService':['RH','SI','RH'], 'service':['SI','', 'RH'], 'weightService':[3,2,6]})
        officeData = pd.DataFrame({'roomID':[0,0,0]})
        spatialTag = ['service' ]
        model, placement = RoomOptimisation( officeData, persoData, ppCatTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 9 )

    def test_resultPPCatMatching0Weight(self) :
        persoData = pd.DataFrame({'inService':['RH','SI','RH'], 'service':['SI','SI', 'RH'], 'weightService':[3,0,6]})
        officeData = pd.DataFrame({'roomID':[0,0,0]})
        spatialTag = ['service' ]
        model, placement = RoomOptimisation( officeData, persoData, ppCatTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 9 )

    def test_resultPPCatMatchingNegWeight(self) :
        persoData = pd.DataFrame({'inService':['RH','SI','RH'], 'service':['SI','SI', 'RH'], 'weightService':[-3,0,-6]})
        officeData = pd.DataFrame({'roomID':[0,0,0]})
        spatialTag = ['service' ]
        model, placement = RoomOptimisation( officeData, persoData, ppCatTag=spatialTag )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), -9 )
    
    def test_resultPPCatMatchingAloneRoom(self) :
        #Check self liking
        persoData = pd.DataFrame({'inService':['RH','SI'], 'service':['SI','SI'], 'weightService':[3,6]})
        officeData = pd.DataFrame({'roomID':[0,1]})
        spatialTag = ['service' ]
        model, placement = RoomOptimisation( officeData, persoData, ppCatTag=spatialTag )
        
        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 6 )

    def test_resultPPBinMatching(self) :
        persoData = pd.DataFrame({'inPhone':[0, 1, 1], 'weightPhone':[-2, 0, 1] })
        officeData = pd.DataFrame({'roomID':[1,0,0]})    
        tag=['phone']
        
        model, placement = RoomOptimisation( officeData, persoData, ppBinTag=tag )
        y = ArrayFromPulpMatrix2D( np.array(placement) )

        self.assertEqual(pulp.LpStatus[model.status], 'Optimal' )
        self.assertEqual(pulp.value(model.objective), 1 )



        self.assertEqual(y[0][0], 1.)



#==========
if __name__ == '__main__':
    unittest.main()
