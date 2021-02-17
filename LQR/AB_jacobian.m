% declare symbols
syms theta xdot thetadot m l g u b M x 

x_doubledot = (2*m*l*thetadot^2*sin(theta)+3*m*g*sin(theta)*cos(theta)+4*(u-b*xdot))/(4*(M+m)-3*m*cos(theta)^2);
theta_doubledot = -3*(m*l*thetadot^2*sin(theta)*cos(theta)+2*((M+m)*g*sin(theta)+(u-b*xdot)*cos(theta)))/(l*(4*(M+m)-3*m*cos(theta)^2));

% jac_A calculate the jacobian matrix of A
jac_A=jacobian([xdot,x_doubledot,theta_doubledot,thetadot],[x,xdot,thetadot,theta]);
% print result
disp(jac_A);

% subsitute variable with real values
% m=0.5 M=0.5 l=0.5 g=9.82 b=1.0 
% x_goal_state=[x, xdot, thetadot, theta]=[0, 0, 0, pi]
jac_A=subs(jac_A,{g,m,M,l,b,theta,thetadot},{9.82,0.5,0.5,0.5,1.0,pi,0});
% print final result of A
disp(jac_A)

%jac_B calculate the jacobian matrix of B
jac_B=jacobian([xdot,(2*m*l*thetadot^2*sin(theta)+3*m*g*sin(theta)*cos(theta)+4*(u-b*xdot))/(4*(M+m)-3*m*cos(theta)^2),-3*(m*l*thetadot^2*sin(theta)*cos(theta)+2*((M+m)*g*sin(theta)+(u-b*xdot)*cos(theta)))/(l*(4*(M+m)-3*m*cos(theta)^2)),thetadot],u);
%print result
disp(jac_B);

% subsitute variable with real values
% m=0.5 M=0.5 l=0.5 g=9.82 b=1.0 
% x_goal_state=[x, xdot, thetadot, theta]=[0, 0, 0, pi]
jac_B=subs(jac_B,{g,m,M,l,b,theta},{9.82,0.5,0.5,0.5,1.0,pi});
%print final result of B
disp(jac_B)