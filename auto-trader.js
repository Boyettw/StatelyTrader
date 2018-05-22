var bittrex = require("node-bittrex-api");
const WebSocket = require("ws");
const wss = new WebSocket.Server({ port: 8080 });
// const ws = new WebSocket('ws://www.host.com/path');
const https = require("https");
var crypto = require("crypto");

//ask for prediction

// returns market name

// query balance for all the markets

// query amount of bitcoin

//take percentage of current balance

//query market for last price and quantity sold

//place buy order for predicted market with the queried amountof btc

// sell anything thats over 2 minutes old

wss.on("connection", function connection(ws) {
  ws.on("message", function incoming(message) {
    console.log("received: %s", message);
  });

  ws.send("something");
});

class AutoTrader {
  constructor() {
    this.apiKey = "d8a28a47ebf44a1cb7760c5428d381b3";
    this.exchangeUrl = "https://bittrex.com";
    this.balanceUrl = "/api/v1.1/account/getbalance";
    this.balancesUrl = "/api/v1.1/account/getbalances";
    this.buyLimitUrl = "/api/v1.1/market/buylimit";
    this.sellLimitUrl = "/api/v1.1/market/selllimit";
    this.openOrdersUrl = "/api/v1.1/market/getopenorders";
    this.marketsUrl = "/api/v1.1/public/getmarkets";
    this.orderBookUrl = "/api/v1.1/public/getorderbook";
    this.marketHistoryUrl = "/api/v1.1/public/getmarkethistory";
    this.amountOfBTC;
    this.secret = "c93a6571297b427f88e24eed9500cce5";
    this.key = "e719603528a544d0b10f902bbb0a8d48";
    bittrex.options({
      apikey: this.apiKey,
      apisecret: this.secret,
      stream: false,
      verbose: true,
      cleartext: false
    });
  }

  fetchPrediction() {}

  queryBalances(apiKey) {

    bittrex.getbalances(function(data) {
      console.log(data);
    });
  }

  queryBalance(apiKey, market) {
    bittrex.getbalance(function(data) {
      console.log(data);
    });
  }

  placeBuyOrder(market, amount) {
    // bittrex.placeBuyOrder(function(data) {
    //   console.log(data);
    // });
  }

  placeSellOrder(market, amount) {}

  fetchLastSellOrder(market) {}
}

console.log("set interval");
let aT = new AutoTrader();
setInterval(() => {
  //check time sell anything older than 2mins
  aT.queryBalances(AutoTrader.apiKey);
}, 5000);

// https
// .get("from python", (resp) => {
//   let data = "";

//   // A chunk of data has been recieved.
//   resp.on("data", chunk => {
//     data += chunk;
//   });

//   // The whole response has been received. Print out the result.
//   resp.on("end", () => {
//     console.log(JSON.parse(data).explanation);
//   });
// })
// .on("error", err => {
//   console.log("Error: " + err.message);
// });
