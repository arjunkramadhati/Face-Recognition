<?php

set_time_limit(0);
$flag1 = 0;
$count = 0;
$fail = 0;
$ip = '127.0.0.1';
$port = 8080;
$token = "959681418:AAGEyAwADSAWIWNLOUB12jA2_MO9uuFx3z8";
$chatid = "929520880";
$operatorID = '922474722';
function sendMessage($chatID, $messaggio, $token) {
    echo "sending message to " . $chatID . "\n";
    $ch = curl_init();
    $url = "https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatID;
    $url = $url . "&text=" . urlencode($messaggio);
    $optArray = array(
            CURLOPT_URL => $url,
            CURLOPT_RETURNTRANSFER => true
    );
    curl_setopt_array($ch, $optArray);
    $result = curl_exec($ch);
    curl_close($ch);
    return $result;
}
/*
 +-------------------------------
 *    @socketcommunicateprocess
 +-------------------------------
 *    @socket_create
 *    @socket_bind
 *    @socket_listen
 *    @socket_accept
 *    @socket_read
 *    @socket_write
 *    @socket_close
 +--------------------------------
 */
if(($sock = socket_create(AF_INET,SOCK_STREAM,SOL_TCP)) < 0) {
    echo "socket_create() Fail to create:".socket_strerror($sock)."\n";
}

if(($ret = socket_bind($sock,$ip,$port)) < 0) {
    echo "socket_bind() Fail to bind:".socket_strerror($ret)."\n";
}

if(($ret = socket_listen($sock,4)) < 0) {
    echo "socket_listen() Fail to listen:".socket_strerror($ret)."\n";
}


do {
    if (($msgsock = socket_accept($sock)) < 0) {
        echo "socket_accept() failed: reason: " . socket_strerror($msgsock) . "\n";
    } else {

        $msg ="Success receive from client！\n";
        socket_write($msgsock, $msg, strlen($msg));
	
	$emp="ARJUN";
        echo "Success\n";
        
        $buf = socket_read($msgsock,8192);
        $buf = strtoupper($buf);
	if((strcmp($buf, $emp))){
	    $fail+=1;
	}else{
        if($flag1 == 1){
        sendMessage($chatid, "worker found", $token);
        sendMessage($operatorID, "worker found", $token);
        $flag1 = 0;
        
        }
        $fail=0;
        }  

        $talkback = "Received Message:$buf\n";
        echo $talkback;


	if($fail == 5){
	    sendMessage($chatid, "worker not found", $token);
            sendMessage($operatorID, "worker not found", $token);
            $fail = 0;
            $flag1 = 1;
	}



    }
    //echo $buf;
    socket_close($msgsock);

} while (true);

socket_close($sock);
?>










































