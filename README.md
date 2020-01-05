# StatelyTrader
Bittrex trading wallet AI, run on your own device for free!


This system is comprised of a number of coupled parts, any downtime can result in fuzzy data (areas where noise(missing data in this case) disturbs the model's manifold)

The Node server listens to bittrex.com and integrates with electron allowing users to trade programatically.

The tensorflow Convolutional-Recurrent Neural Net evaluates different markets "hotness" or likelihood to increase in demand/value. It does so by using many recurrent neural nets which are trained on different offsets into the future (120000 millisecond increments up to 1200000 milliseconds) and handing the softmax outputs 
(1d arrays of softmax floats to form (1(last 5 values in eachmarket + num_rnns) * num_markets) to a CNN which selects optimal markets supplied via softmax from percentile scores generated from evaluate.py due to the aforementioned softmax outputs.

The Node server listens for contact from the python AI upon startup, then begins sending the AI trade data from the supplied market list. On continued contact from the server the Node server authenticates with bittrex using the user's credentials and queries the balances on the account. On reception of a buy market msg from the python process we begin posting sell orders on all markets that are not the the top 5 votes from the softmax output and a buy order on the msged market for the last recieved market price from the market sent from the AI. This will require keeping running sets of data on hand in a sorted fashion(probably B tree).

The AI loads up its tensor variables/states and begins predicting when enough data is loaded (history length)
When predicted market changes the node server is notified of the new priorities.


