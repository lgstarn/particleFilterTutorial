function brownianBridgeTest()
    disp('Now running bridge test one...')
    brownianBridge(5, 100, 1, .1);
    disp('Done.  Press any key to continue.')
    pause
    disp('Now running bridge test two...')
    brownianBridge(5, 100, 1, 1);
    disp('Done.  Press any key to continue.')
    pause
    disp('Now running bridge test two...')
    brownianBridge(5, 100, 1, 10);
    disp('Done.  Press any key to continue.')
    pause
end

