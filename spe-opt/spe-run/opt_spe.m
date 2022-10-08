%% plot the single optimal results
close all
xx=1:200;
ind=max(xx);
colorlist=[[0,114,189]/255; [217,83,25]/255; [237,177,32]/255; [126,47,142]/255];
bia=zeros(1,ind);
for kk=1:ind
    img=paramHist(kk,:,:);
    bia(kk)=mean(mean(abs(img).*(2-abs(img))));
end

figure(1)
[hAx,hLine1,hLine2]=plotyy(xx,fomHist(1:ind),xx,-bia(1:ind));
xlabel('Epochs')
ylabel(hAx(1),'Coupling efficiency') 
ylabel(hAx(2),'Discreteness')
ylim(hAx(1),[0,1])
ylim(hAx(2),[-1,0])
xlim(hAx(1),[0,ind])
xlim(hAx(2),[0,ind])
hLine1.LineWidth=1;
hLine2.LineWidth=1;
set(hAx(1),'YColor',colorlist(3,:))
set(hLine1,'Color',colorlist(3,:))
set(hAx(2),'YColor',colorlist(4,:));
set(hLine2,'Color',colorlist(4,:))
set(hAx,'FontSize',10,'FontName','Times New Roman')
set(hAx,'YTickMode','auto')
set(gcf,'Position',[0,0,530,450])

figure(3)
img=paramHist(ind,:,:);
img=reshape(img,101,101)';
img=(img+1)/2;
imshow(1-img)
set(gcf,'Position',[0,0,300,300])

fomHist(ind)
bia(ind)