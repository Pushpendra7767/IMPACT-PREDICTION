<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<title>ML API</title>
<style>
body {font-family: Arial, Helvetica, sans-serif;}
form {border: 3px solid #f1f1f1;}



body { 
	width: 60%;
	height:120%;
	font-family: 'Open Sans', sans-serif;
	background: #092756;
	color: #fff;
	font-size: 18px;
	text-align:left;
	letter-spacing:1.2px;
	background: -moz-radial-gradient(0% 100%, ellipse cover, rgba(104,128,138,.4) 10%,rgba(138,114,76,0) 40%),-moz-linear-gradient(top,  rgba(57,173,219,.25) 0%, rgba(42,60,87,.4) 100%), -moz-linear-gradient(-45deg,  #670d10 0%, #092756 100%);
	background: -webkit-radial-gradient(0% 100%, ellipse cover, rgba(104,128,138,.4) 10%,rgba(138,114,76,0) 40%), -webkit-linear-gradient(top,  rgba(57,173,219,.25) 0%,rgba(42,60,87,.4) 100%), -webkit-linear-gradient(-45deg,  #670d10 0%,#092756 100%);
	background: -o-radial-gradient(0% 100%, ellipse cover, rgba(104,128,138,.4) 10%,rgba(138,114,76,0) 40%), -o-linear-gradient(top,  rgba(57,173,219,.25) 0%,rgba(42,60,87,.4) 100%), -o-linear-gradient(-45deg,  #670d10 0%,#092756 100%);
	background: -ms-radial-gradient(0% 100%, ellipse cover, rgba(104,128,138,.4) 10%,rgba(138,114,76,0) 40%), -ms-linear-gradient(top,  rgba(57,173,219,.25) 0%,rgba(42,60,87,.4) 100%), -ms-linear-gradient(-45deg,  #670d10 0%,#092756 100%);
	background: -webkit-radial-gradient(0% 100%, ellipse cover, rgba(104,128,138,.4) 10%,rgba(138,114,76,0) 40%), linear-gradient(to bottom,  rgba(57,173,219,.25) 0%,rgba(42,60,87,.4) 100%), linear-gradient(135deg,  #670d10 0%,#092756 100%);
	filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#3E1D6D', endColorstr='#092756',GradientType=1 );

}

input[type=text], input[type=text],input[type=text],input[type=text],input[type=text],input[type=text],input[type=text],input[type=text],input[type=text],input[type=text],input[type=text],input[type=text],input[type=text] {
    width: 100%; 
	margin-bottom: 10px; 
	background: rgba(0,0,0,0.3);
	border: none;
	
	outline: none;
	padding: 10px;
	font-size: 13px;
	color: #fff;
	text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
	border: 1px solid rgba(0,0,0,0.3);
	border-radius: 4px;
	box-shadow: inset 0 -5px 45px rgba(100,100,100,0.2), 0 1px 1px rgba(255,255,255,0.2);
	-webkit-transition: box-shadow .5s ease;
	-moz-transition: box-shadow .5s ease;
	-o-transition: box-shadow .5s ease;
	-ms-transition: box-shadow .5s ease;
	transition: box-shadow .5s ease;
}



button {
  background-color: #4CAF50;
  color: white;
  padding: 14px 20px;
  margin: 8px 0;
  border: none;
  cursor: pointer;
  width: 100%;
}

button:hover {
  opacity: 0.8;
}

.cancelbtn {
  width: auto;
  padding: 10px 18px;
  background-color: #f44336;
}

.imgcontainer {
  text-align: center;
  margin: 24px 0 12px 0;
}

img.avatar {
  width: 40%;
  border-radius: 50%;
}

.container {
  padding: 16px;
}

span.psw {
  float: right;
  padding-top: 16px;
}

/* Change styles for span and cancel button on extra small screens */
@media screen and (max-width: 300px) {
  span.psw {
     display: block;
     float: none;
  }
  .cancelbtn {
     width: 100%;
  }
}
</style>
</head>
<body>

<h2>IMPACT PREDICTION</h2>

<form action="{{ url_for('predict')}}"method="post">
  <div class="imgcontainer">
    <img src="C:\Users\ACER\Desktop\project\my.png" alt="Avatar" class="avatar">
  </div>

  <div class="container">
    <label for="uname"><b>Username</b></label>
    <input type="text" placeholder="Enter Username" name="uname" required>

    <label for="psw"><b>Password</b></label>
    <input type="text" placeholder="Enter Password" name="psw" required>
    
    <label for="psw1"><b>Password1</b></label>
    <input type="text" placeholder="Enter Password1" name="psw1" required>
    
    <label for="psw2"><b>Password2</b></label>
    <input type="text" placeholder="Enter Password2" name="psw2" required>
    
    <label for="psw3"><b>Password3</b></label>
    <input type="text" placeholder="Enter Password3" name="psw3" required>
    
    <label for="psw4"><b>Password4</b></label>
    <input type="text" placeholder="Enter Password4" name="psw4" required>
    
    <label for="psw5"><b>Password5</b></label>
    <input type="text" placeholder="Enter Password5" name="psw5" required>
    
    <label for="psw6"><b>Password6</b></label>
    <input type="text" placeholder="Enter Password6" name="psw6" required>
    
    <label for="psw7"><b>Password7</b></label>
    <input type="text" placeholder="Enter Password7" name="psw7" required>
    
    <label for="psw8"><b>Password8</b></label>
    <input type="text" placeholder="Enter Password8" name="psw8" required>
    
    <label for="psw9"><b>Password9</b></label>
    <input type="text" placeholder="Enter Password9" name="psw9" required>
    
    <label for="psw10"><b>Password10</b></label>
    <input type="text" placeholder="Enter Password10" name="psw10" required>
    
    <label for="psw11"><b>Password11</b></label>
    <input type="text" placeholder="Enter Password11" name="psw11" required>
    
        
   <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
   </form>
   <br>
   <br>
   {{ prediction_text }}

    
  </div>

  



</body>
</html>

##############################################
###################################################################################
<!DOCTYPE html>
<html >
<!--From https://codepen.io/frytyler/pen/EGdtg-->
<head>
  <meta charset="UTF-8">
  <title>ML API</title>
  <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Arimo' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Hind:300' rel='stylesheet' type='text/css'>
<link href='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>
<link rel="stylesheet" href="{{ url_for('static', filename='mstyle.css') }}">
  
</head>

<body>
 <div class="login">
	<h1>Predict Impact Analysis</h1>

     <!-- Main Input For Receiving Query to our ML -->
    <form action="{{ url_for('predict')}}"method="post">
    	<input type="text" name="ID" placeholder="ID" required="required" />
        <input type="text" name="ID_status" placeholder="ID STATUS" required="required" />
		<input type="text" name="count_reassign" placeholder="COUNT REASSIGN" required="required" />
		<input type="text" name="count_updated" placeholder="COUNT UPDATED" required="required" />
		<input type="text" name="ID_caller" placeholder="ID CALLER" required="required" />
		<input type="text" name="opened_by" placeholder="OPENED BY" required="required" />
		<input type="text" name="Created_by" placeholder="CREATED BY" required="required" />
		<input type="text" name="updated_by" placeholder="UPDATED BY" required="required" />
		<input type="text" name="location" placeholder="LOCATION" required="required" />
		<input type="text" name="category_ID" placeholder="CATEGORY ID" required="required" />
		<input type="text" name="user_symptom" placeholder="USER SYMPTOM" required="required" />
		<input type="text" name="Support_group" placeholder="SUPPORT GROUP" required="required" />
		<input type="text" name="support_incharge" placeholder="SUPPORT INCHARGE" required="required" />
		

        <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
    </form>

   <br>
   <br>
   {{ prediction_text }}

 </div>


</body>
</html>


