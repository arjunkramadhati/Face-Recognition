<?php
error_reporting(E_ALL);
set_time_limit(0);
echo "TCP/IP Connection\n";

$port = 36500;
$ip = "192.168.123.146
..............";

/*
 +-------------------------------
 *    @socketconectionprocess
 +-------------------------------
 *    @socket_create
 *    @socket_connect
 *    @socket_write
 *    @socket_read
 *    @socket_close
 +--------------------------------
 */

$socket = socket_create(AF_INET, SOCK_STREAM, SOL_TCP);
if ($socket < 0) {
    echo "socket_create() failed. reason: " . socket_strerror($socket) . "\n";
}else {
    echo "OK.\n";
}

echo "Try to connect '$ip' Port '$port'...\n";
$result = socket_connect($socket, $ip, $port);
if ($result < 0) {
    echo "socket_connect() failed.\nReason: ($result) " . socket_strerror($result) . "\n";
}else {
    echo "Connect OK\n";
}


$myfile = fopen("/home/sigsenz/Desktop/FaceRec-master_new_FR/status.txt", "r") or die("Unable to open file!");
$in = fread($myfile,filesize("/home/sigsenz/Desktop/FaceRec-master_new_FR/status.txt"));
fclose($myfile);
$out = '';

if(!socket_write($socket, $in, strlen($in))) {
    echo "socket_write() failed. reason: " . socket_strerror($socket) . "\n";
}else {
    echo "Send Message to Server Successfully!\n";
    echo "Send Information:<font color='red'>$in</font> <br>";
}

while($out = socket_read($socket, 8192)) {
    echo "Receive Server Return Message Successfully!\n";
    echo "Received Message:",$out;
}


echo "Turn Off Socket...\n";
socket_close($socket);
echo "Turn Off OK\n";
?>
