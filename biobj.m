% % % read data
% X = readmatrix('data/data_all2.csv');
% 
% % % model inputs
% [I,K] = size(X);
% nj = 20;
% r = 3;
% 
% V = cell(K, 1);
% C = [];
% cnt = 0;
% for k = 1:K
%     V_k = {};
%     for i = 1:I
%         if X(i, k) > 0 && X(i, k) <= r
%             cnt = cnt + 1;
%             C = [C; [i, k]];
%             V_k = [V_k, i];
%         end
%     end
%     V{k} = V_k;
% end
% 
% W = zeros(cnt, I);
% for ik = 1:cnt
%     i = C(ik, 1);
%     k = C(ik, 2);
%     for j = 1:I
%         if X(i, k) == 0 || X(j, k) == 0
%             W(ik, j) = 50;
%         else
%             W(ik, j) = 1 / X(i, k) + 1 / X(j, k);
%         end
%     end
% end
% 
% T = ones(cnt, K);
% for ik = 1:cnt
%     i = C(ik, 1);
%     k = C(ik, 2);
%     for j = 1:I
%         if i == j || any(ismember([V{k}{:}], j))
%             T(ik, j) = 0;
%         end
%     end
% end
% 
% X0 = zeros(cnt, 1);
% for ik = 1:cnt
%     i = C(ik, 1);
%     k = C(ik, 2);
%     X0(ik) = X(i, k);
% end
% 
% 
% 
% % decision variables
% prob = optimproblem();
% thetakij = optimvar('thetakij', cnt, I, Type='continuous', LowerBound=0, UpperBound=1);
%  
% 
% % constraints
% cons1 = optimconstr(cnt, 1);
% cons2 = optimconstr(cnt, 1);
% temp1 = 0;
% temp2 = 0;
% for ik = 1:cnt
%     i = C(ik, 1);
%     k = C(ik, 2);
%     cons1(ik) = sum(thetakij(ik,:)) == 1;
%     cons2(ik) = thetakij(ik,i) + sum(T(ik,:) .* thetakij(ik,:)) == 1;
%     temp1 = temp1 + sum(T(ik,:) .* thetakij(ik,:)) * X0(ik);
%     temp2 = temp2 + sum(W(ik,:) .* thetakij(ik,:)) * X0(ik);
% end
% prob.Constraints.cons1 = cons1;
% prob.Constraints.cons2 = cons2;
% prob.Objective.obj1 = -temp1;
% prob.Objective.obj2 = temp2;
% 
% cons3 = optimconstr(I, 1);
% for j = 1:I
%     cons3(j) = sum(T(:,j) .* thetakij(:,j) .* X0) <= nj;
% end
% prob.Constraints.cons3 = cons3;



rng default % For reproducibility
% opts = optimoptions("gamultiobj",PlotFcn="gaplotpareto",PopulationSize=5e2,...
%     MutationFcn=@mutationgaussian);
% [sol,fval,exitflag,output] = solve(prob,Options=opts);
% 
% paretoplot(sol)