var bittrex = require('node-bittrex-api');
const MongoClient = require('mongodb').MongoClient;
let subscriptions = undefined;

async function start() {
	var startTime = new Date().getTime;
    let url = 'mongodb://10.0.0.88:27017';
    let client = await MongoClient.connect(url);
    let db = client.db('markets');
    let collections = await db.collections();
    subscriptions = [];
    for (let i = 0; i < collections.length; i++) {
        subscriptions.push(collections[i].s.name);
        console.log(collections[i].s.name);
    }
    bittrex.options({
        'verbose': true,
    });

    bittrex.options({
        verbose: true,
        websockets: {
            onConnect: function () {
                console.log('onConnect fired');
                bittrex.websockets.subscribe(subscriptions, function (msg, client) {//subscriptions
                    //console.log(data);
                    if (msg.M === 'updateExchangeState') {
                        msg.A.forEach(function (data) {
                            //console.log('Market Update for ' + data.MarketName);
                            //console.log(data);
                            let filled = data.Fills;
                            let trade, epoch, orderType, rate, quantity;
							let trades = [];
                            for(let i = 0; i < filled.length; i++){
                                trade = filled[i];
                                epoch = new Date(trade.TimeStamp).getTime();
                                orderType = trade.OrderType === 'BUY'? 1 : -1;
                                rate = trade.Rate;
                                quantity = trade.Quantity;
                                //console.log(filled[i]);
								trades.push({epoch: epoch, orderType: orderType, rate: rate, quantity: quantity});
                            }
							if(trades.length !== 0){
								db.collection(data.MarketName).insertMany(trades);
								console.log("Inserted trades for: " + data.MarketName);
								if((epoch - startTime) > 18000000){
									exit();
								}
							}
                        });
                    }
                });
            },
        },
    });

    console.log('Connecting ....');
    bittrex.websockets.client(function (client) {
        // connected - you can do any one-off connection events here
        //
        // Note: Reoccuring events like listen() and subscribe() should be done
        // in onConnect so that they are fired during a reconnection event.
        console.log('Connected');
    });
}

start();