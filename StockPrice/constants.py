class SelectOptions:
    def __init__(self, value, text):
        self.value = value
        self.text = text


intervals = [
    SelectOptions("15m", "15 Minutes"),
    SelectOptions("1h", "1 Hour"),
    SelectOptions("6h", "6 Hours", ),
    SelectOptions("12h", "12 Hours"),
    SelectOptions("1d", "1 Day"),
    # SelectOptions("1w", "1 Week"),
    # SelectOptions("1M", "1 Month"),
]

features = [
    SelectOptions("Close", "Close"),
    SelectOptions("ROC", "Price of change"),
    SelectOptions("MA", "Moving Average"),
    # SelectOptions("BB", "Bollinger Band")
]
