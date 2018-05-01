"""
Load saved model from disk,
*if none supplied build from Train.py yourself live (one by one aka cut minibatch size by 1 dimension and slice).
*Load values into a linked list that you set into a numpy array every train/predict slice,
*one for features you maintain up to 6000 epoch back + 101 additional trades from that point on.
*Train two RNNs at a time one along sell track 300 epoch into the future,
*and another 6000 into the past with 101 additional.

Plugs into Bittrex websocket API and loads only 6000 epoch + the additional 101 trades on linked list.
Converts linkedlist into array and calls predict on trained model.
Calls account api to access coin balances pairing them with market.
    #eventually this will
Call Bittrex limit sell API and post sells listening to 30s sell RNN.
Call bittrex limit buy API and post buys when 30s does not classify and when 6min classifies labeled market as buy.
We will need raw data on the "strength" weight of the outputs before softmax,
but could potentially quantify how strong a normal distribution of classification is.

"""