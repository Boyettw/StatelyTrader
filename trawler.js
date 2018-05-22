var bittrex = require('node-bittrex-api');
const MongoClient = require('mongodb').MongoClient;
let subscriptions = undefined;
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.broadcast = function broadcast(data) {
    wss.clients.forEach(function each(client) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(data);
      }
    });
  };

  
wss.on('connection', function connection(ws) {
    ws.on('message', function incoming(data) {
      // Broadcast to everyone else.
      wss.clients.forEach(function each(client) {
        if (client !== ws && client.readyState === WebSocket.OPEN) {
          client.send(data);
        }
      });
    });
  });

async function start() {
	var startTime = new Date().getTime;
    let url = 'mongodb://127.0.0.1:27017';
    let client = await MongoClient.connect(url);
    let db = client.db('markets');
    let collections = await db.collections();
    subscriptions = [];
    for (let i = 0; i < collections.length; i++) {
        subscriptions.push(collections[i].s.name);
        console.log(collections[i].s.name);
    }


    bittrex.options({
        verbose: true,
        websockets: {
            onConnect: () => {
                console.log('onConnect fired');
                bittrex.websockets.subscribe(subscriptions, function (msg, client) {//subscriptions
                    //console.log(data);

                    switch(msg.M){
                        case 'updateExchangeState':
                            msg.A.forEach((data) => {
                                //console.log('Market Update for ' + data.MarketName);
                                console.log(data);
                                let filled = data.Fills;
                                let trade, epoch, orderType, rate, quantity;
                                let trades = [];
                                for(let i = 0; i < filled.length; i++){
                                    trade = filled[i];
                                    epoch = new Date(trade.TimeStamp).getTime();
                                    orderType = trade.OrderType === 'BUY'? 1 : -1;
                                    rate = trade.Rate;
                                    quantity = trade.Quantity;

                                    wss.broadcast({epoch: epoch, orderType: orderType, rate: rate, quantity: quantity});
                                    // trades.push({epoch: epoch, orderType: orderType, rate: rate, quantity: quantity});
                                }
                                // if(trades.length !== 0){
                                //     db.collection(data.MarketName).insertMany(trades);
                                //     console.log("Inserted trades for: " + data.MarketName);
                                //     if((epoch - startTime) > 18000000){
                                //         exit();
                                //     }
                                // }

                            });
                        break;
                    }              
                });
            },
        },
    });

    console.log('Connecting ....');
    bittrex.websockets.client(client => {
        // connected - you can do any one-off connection events here
        //
        // Note: Reoccuring events like listen() and subscribe() should be done
        // in onConnect so that they are fired during a reconnection event.
        console.log('Connected');
    },error => {
        console.error(error);
    });
}

start();