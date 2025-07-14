import robin_stocks.robinhood as r

username = "your_robinhood_username"
password = "your_robinhood_password"

# go to local history
# Login
r.authentication.login(username, password)

# Fetch your positions
positions = r.profiles.load_account_profile()
print(positions)

# Or your stock positions
stock_positions = r.build_holdings()
print(stock_positions)

# Logout
r.authentication.logout()