
_tlang_cache//tmp0001.cpp.so:     file format elf64-x86-64


Disassembly of section .init:

00000000000006a8 <_init>:
 6a8:	48 83 ec 08          	sub    rsp,0x8
 6ac:	48 8b 05 35 19 20 00 	mov    rax,QWORD PTR [rip+0x201935]        # 201fe8 <__gmon_start__>
 6b3:	48 85 c0             	test   rax,rax
 6b6:	74 02                	je     6ba <_init+0x12>
 6b8:	ff d0                	call   rax
 6ba:	48 83 c4 08          	add    rsp,0x8
 6be:	c3                   	ret    

Disassembly of section .plt:

00000000000006c0 <.plt>:
 6c0:	ff 35 42 19 20 00    	push   QWORD PTR [rip+0x201942]        # 202008 <_GLOBAL_OFFSET_TABLE_+0x8>
 6c6:	ff 25 44 19 20 00    	jmp    QWORD PTR [rip+0x201944]        # 202010 <_GLOBAL_OFFSET_TABLE_+0x10>
 6cc:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

00000000000006d0 <memset@plt>:
 6d0:	ff 25 42 19 20 00    	jmp    QWORD PTR [rip+0x201942]        # 202018 <memset@GLIBC_2.2.5>
 6d6:	68 00 00 00 00       	push   0x0
 6db:	e9 e0 ff ff ff       	jmp    6c0 <.plt>

00000000000006e0 <__cxa_atexit@plt>:
 6e0:	ff 25 3a 19 20 00    	jmp    QWORD PTR [rip+0x20193a]        # 202020 <__cxa_atexit@GLIBC_2.2.5>
 6e6:	68 01 00 00 00       	push   0x1
 6eb:	e9 d0 ff ff ff       	jmp    6c0 <.plt>

00000000000006f0 <_ZdlPv@plt>:
 6f0:	ff 25 32 19 20 00    	jmp    QWORD PTR [rip+0x201932]        # 202028 <_ZdlPv@GLIBCXX_3.4>
 6f6:	68 02 00 00 00       	push   0x2
 6fb:	e9 c0 ff ff ff       	jmp    6c0 <.plt>

0000000000000700 <_Znwm@plt>:
 700:	ff 25 2a 19 20 00    	jmp    QWORD PTR [rip+0x20192a]        # 202030 <_Znwm@GLIBCXX_3.4>
 706:	68 03 00 00 00       	push   0x3
 70b:	e9 b0 ff ff ff       	jmp    6c0 <.plt>

0000000000000710 <_ZNSt8ios_base4InitC1Ev@plt>:
 710:	ff 25 22 19 20 00    	jmp    QWORD PTR [rip+0x201922]        # 202038 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
 716:	68 04 00 00 00       	push   0x4
 71b:	e9 a0 ff ff ff       	jmp    6c0 <.plt>

Disassembly of section .plt.got:

0000000000000720 <__cxa_finalize@plt>:
 720:	ff 25 b2 18 20 00    	jmp    QWORD PTR [rip+0x2018b2]        # 201fd8 <__cxa_finalize@GLIBC_2.2.5>
 726:	66 90                	xchg   ax,ax

Disassembly of section .text:

0000000000000730 <_GLOBAL__sub_I_tmp0001.cpp>:
     730:	53                   	push   rbx
     731:	48 8d 1d 11 19 20 00 	lea    rbx,[rip+0x201911]        # 202049 <_ZStL8__ioinit>
     738:	48 89 df             	mov    rdi,rbx
     73b:	e8 d0 ff ff ff       	call   710 <_ZNSt8ios_base4InitC1Ev@plt>
     740:	48 8b 3d b1 18 20 00 	mov    rdi,QWORD PTR [rip+0x2018b1]        # 201ff8 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
     747:	48 8d 15 f2 18 20 00 	lea    rdx,[rip+0x2018f2]        # 202040 <__dso_handle>
     74e:	48 89 de             	mov    rsi,rbx
     751:	5b                   	pop    rbx
     752:	e9 89 ff ff ff       	jmp    6e0 <__cxa_atexit@plt>
     757:	66 0f 1f 84 00 00 00 	nop    WORD PTR [rax+rax*1+0x0]
     75e:	00 00 

0000000000000760 <deregister_tm_clones>:
     760:	48 8d 3d e1 18 20 00 	lea    rdi,[rip+0x2018e1]        # 202048 <_edata>
     767:	55                   	push   rbp
     768:	48 8d 05 d9 18 20 00 	lea    rax,[rip+0x2018d9]        # 202048 <_edata>
     76f:	48 39 f8             	cmp    rax,rdi
     772:	48 89 e5             	mov    rbp,rsp
     775:	74 19                	je     790 <deregister_tm_clones+0x30>
     777:	48 8b 05 62 18 20 00 	mov    rax,QWORD PTR [rip+0x201862]        # 201fe0 <_ITM_deregisterTMCloneTable>
     77e:	48 85 c0             	test   rax,rax
     781:	74 0d                	je     790 <deregister_tm_clones+0x30>
     783:	5d                   	pop    rbp
     784:	ff e0                	jmp    rax
     786:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
     78d:	00 00 00 
     790:	5d                   	pop    rbp
     791:	c3                   	ret    
     792:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
     796:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
     79d:	00 00 00 

00000000000007a0 <register_tm_clones>:
     7a0:	48 8d 3d a1 18 20 00 	lea    rdi,[rip+0x2018a1]        # 202048 <_edata>
     7a7:	48 8d 35 9a 18 20 00 	lea    rsi,[rip+0x20189a]        # 202048 <_edata>
     7ae:	55                   	push   rbp
     7af:	48 29 fe             	sub    rsi,rdi
     7b2:	48 89 e5             	mov    rbp,rsp
     7b5:	48 c1 fe 03          	sar    rsi,0x3
     7b9:	48 89 f0             	mov    rax,rsi
     7bc:	48 c1 e8 3f          	shr    rax,0x3f
     7c0:	48 01 c6             	add    rsi,rax
     7c3:	48 d1 fe             	sar    rsi,1
     7c6:	74 18                	je     7e0 <register_tm_clones+0x40>
     7c8:	48 8b 05 21 18 20 00 	mov    rax,QWORD PTR [rip+0x201821]        # 201ff0 <_ITM_registerTMCloneTable>
     7cf:	48 85 c0             	test   rax,rax
     7d2:	74 0c                	je     7e0 <register_tm_clones+0x40>
     7d4:	5d                   	pop    rbp
     7d5:	ff e0                	jmp    rax
     7d7:	66 0f 1f 84 00 00 00 	nop    WORD PTR [rax+rax*1+0x0]
     7de:	00 00 
     7e0:	5d                   	pop    rbp
     7e1:	c3                   	ret    
     7e2:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
     7e6:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
     7ed:	00 00 00 

00000000000007f0 <__do_global_dtors_aux>:
     7f0:	80 3d 51 18 20 00 00 	cmp    BYTE PTR [rip+0x201851],0x0        # 202048 <_edata>
     7f7:	75 2f                	jne    828 <__do_global_dtors_aux+0x38>
     7f9:	48 83 3d d7 17 20 00 	cmp    QWORD PTR [rip+0x2017d7],0x0        # 201fd8 <__cxa_finalize@GLIBC_2.2.5>
     800:	00 
     801:	55                   	push   rbp
     802:	48 89 e5             	mov    rbp,rsp
     805:	74 0c                	je     813 <__do_global_dtors_aux+0x23>
     807:	48 8b 3d 32 18 20 00 	mov    rdi,QWORD PTR [rip+0x201832]        # 202040 <__dso_handle>
     80e:	e8 0d ff ff ff       	call   720 <__cxa_finalize@plt>
     813:	e8 48 ff ff ff       	call   760 <deregister_tm_clones>
     818:	c6 05 29 18 20 00 01 	mov    BYTE PTR [rip+0x201829],0x1        # 202048 <_edata>
     81f:	5d                   	pop    rbp
     820:	c3                   	ret    
     821:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
     828:	f3 c3                	repz ret 
     82a:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]

0000000000000830 <frame_dummy>:
     830:	55                   	push   rbp
     831:	48 89 e5             	mov    rbp,rsp
     834:	5d                   	pop    rbp
     835:	e9 66 ff ff ff       	jmp    7a0 <register_tm_clones>
     83a:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]

0000000000000840 <create_data_structure>:
     840:	53                   	push   rbx
     841:	bf 00 20 00 10       	mov    edi,0x10002000
     846:	e8 b5 fe ff ff       	call   700 <_Znwm@plt>
     84b:	48 89 c3             	mov    rbx,rax
     84e:	31 f6                	xor    esi,esi
     850:	ba 00 20 00 10       	mov    edx,0x10002000
     855:	48 89 c7             	mov    rdi,rax
     858:	e8 73 fe ff ff       	call   6d0 <memset@plt>
     85d:	48 89 d8             	mov    rax,rbx
     860:	5b                   	pop    rbx
     861:	c3                   	ret    
     862:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
     869:	00 00 00 
     86c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

0000000000000870 <release_data_structure>:
     870:	48 85 ff             	test   rdi,rdi
     873:	74 05                	je     87a <release_data_structure+0xa>
     875:	e9 76 fe ff ff       	jmp    6f0 <_ZdlPv@plt>
     87a:	c3                   	ret    
     87b:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]

0000000000000880 <func000001>:
     880:	48 81 ec 58 01 00 00 	sub    rsp,0x158
     887:	4c 8b 8c 24 60 01 00 	mov    r9,QWORD PTR [rsp+0x160]
     88e:	00 
     88f:	49 8d 89 00 00 00 10 	lea    rcx,[r9+0x10000000]
     896:	48 c7 c2 00 00 c0 ff 	mov    rdx,0xffffffffffc00000
     89d:	c4 e2 79 18 05 ae 09 	vbroadcastss xmm0,DWORD PTR [rip+0x9ae]        # 1254 <_fini+0x38>
     8a4:	00 00 
     8a6:	c5 f8 29 84 24 20 01 	vmovaps XMMWORD PTR [rsp+0x120],xmm0
     8ad:	00 00 
     8af:	c4 e2 79 18 05 a0 09 	vbroadcastss xmm0,DWORD PTR [rip+0x9a0]        # 1258 <_fini+0x3c>
     8b6:	00 00 
     8b8:	c5 f8 29 84 24 10 01 	vmovaps XMMWORD PTR [rsp+0x110],xmm0
     8bf:	00 00 
     8c1:	c4 e2 79 18 05 92 09 	vbroadcastss xmm0,DWORD PTR [rip+0x992]        # 125c <_fini+0x40>
     8c8:	00 00 
     8ca:	c5 f8 29 84 24 00 01 	vmovaps XMMWORD PTR [rsp+0x100],xmm0
     8d1:	00 00 
     8d3:	c4 e2 79 18 05 84 09 	vbroadcastss xmm0,DWORD PTR [rip+0x984]        # 1260 <_fini+0x44>
     8da:	00 00 
     8dc:	c5 f8 29 84 24 f0 00 	vmovaps XMMWORD PTR [rsp+0xf0],xmm0
     8e3:	00 00 
     8e5:	c4 e2 79 18 05 76 09 	vbroadcastss xmm0,DWORD PTR [rip+0x976]        # 1264 <_fini+0x48>
     8ec:	00 00 
     8ee:	c5 f8 29 84 24 e0 00 	vmovaps XMMWORD PTR [rsp+0xe0],xmm0
     8f5:	00 00 
     8f7:	c4 e2 79 18 05 68 09 	vbroadcastss xmm0,DWORD PTR [rip+0x968]        # 1268 <_fini+0x4c>
     8fe:	00 00 
     900:	c5 f8 29 84 24 d0 00 	vmovaps XMMWORD PTR [rsp+0xd0],xmm0
     907:	00 00 
     909:	c4 e2 79 18 05 5a 09 	vbroadcastss xmm0,DWORD PTR [rip+0x95a]        # 126c <_fini+0x50>
     910:	00 00 
     912:	c5 f8 29 84 24 c0 00 	vmovaps XMMWORD PTR [rsp+0xc0],xmm0
     919:	00 00 
     91b:	c4 e2 79 18 05 4c 09 	vbroadcastss xmm0,DWORD PTR [rip+0x94c]        # 1270 <_fini+0x54>
     922:	00 00 
     924:	c5 f8 29 84 24 b0 00 	vmovaps XMMWORD PTR [rsp+0xb0],xmm0
     92b:	00 00 
     92d:	c4 e2 79 18 05 3e 09 	vbroadcastss xmm0,DWORD PTR [rip+0x93e]        # 1274 <_fini+0x58>
     934:	00 00 
     936:	c5 f8 29 84 24 a0 00 	vmovaps XMMWORD PTR [rsp+0xa0],xmm0
     93d:	00 00 
     93f:	c4 e2 79 18 05 30 09 	vbroadcastss xmm0,DWORD PTR [rip+0x930]        # 1278 <_fini+0x5c>
     946:	00 00 
     948:	c5 f8 29 84 24 90 00 	vmovaps XMMWORD PTR [rsp+0x90],xmm0
     94f:	00 00 
     951:	c4 e2 79 18 05 22 09 	vbroadcastss xmm0,DWORD PTR [rip+0x922]        # 127c <_fini+0x60>
     958:	00 00 
     95a:	c5 f8 29 84 24 80 00 	vmovaps XMMWORD PTR [rsp+0x80],xmm0
     961:	00 00 
     963:	c4 e2 79 18 05 14 09 	vbroadcastss xmm0,DWORD PTR [rip+0x914]        # 1280 <_fini+0x64>
     96a:	00 00 
     96c:	c5 f8 29 44 24 70    	vmovaps XMMWORD PTR [rsp+0x70],xmm0
     972:	c4 e2 79 18 05 09 09 	vbroadcastss xmm0,DWORD PTR [rip+0x909]        # 1284 <_fini+0x68>
     979:	00 00 
     97b:	c5 f8 29 44 24 60    	vmovaps XMMWORD PTR [rsp+0x60],xmm0
     981:	c4 e2 79 18 05 fe 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8fe]        # 1288 <_fini+0x6c>
     988:	00 00 
     98a:	c5 f8 29 44 24 50    	vmovaps XMMWORD PTR [rsp+0x50],xmm0
     990:	c4 e2 79 18 05 f3 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8f3]        # 128c <_fini+0x70>
     997:	00 00 
     999:	c5 f8 29 44 24 40    	vmovaps XMMWORD PTR [rsp+0x40],xmm0
     99f:	c4 e2 79 18 05 e8 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8e8]        # 1290 <_fini+0x74>
     9a6:	00 00 
     9a8:	c5 f8 29 44 24 30    	vmovaps XMMWORD PTR [rsp+0x30],xmm0
     9ae:	c4 e2 79 18 05 dd 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8dd]        # 1294 <_fini+0x78>
     9b5:	00 00 
     9b7:	c5 f8 29 44 24 20    	vmovaps XMMWORD PTR [rsp+0x20],xmm0
     9bd:	c4 e2 79 18 05 d2 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8d2]        # 1298 <_fini+0x7c>
     9c4:	00 00 
     9c6:	c5 f8 29 44 24 10    	vmovaps XMMWORD PTR [rsp+0x10],xmm0
     9cc:	c4 e2 79 18 05 c7 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8c7]        # 129c <_fini+0x80>
     9d3:	00 00 
     9d5:	c5 f8 29 04 24       	vmovaps XMMWORD PTR [rsp],xmm0
     9da:	c4 e2 79 18 05 bd 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8bd]        # 12a0 <_fini+0x84>
     9e1:	00 00 
     9e3:	c5 f8 29 44 24 f0    	vmovaps XMMWORD PTR [rsp-0x10],xmm0
     9e9:	c4 e2 79 18 05 b2 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8b2]        # 12a4 <_fini+0x88>
     9f0:	00 00 
     9f2:	c5 f8 29 44 24 e0    	vmovaps XMMWORD PTR [rsp-0x20],xmm0
     9f8:	c4 e2 79 18 05 a7 08 	vbroadcastss xmm0,DWORD PTR [rip+0x8a7]        # 12a8 <_fini+0x8c>
     9ff:	00 00 
     a01:	c5 f8 29 44 24 d0    	vmovaps XMMWORD PTR [rsp-0x30],xmm0
     a07:	c4 e2 79 18 05 9c 08 	vbroadcastss xmm0,DWORD PTR [rip+0x89c]        # 12ac <_fini+0x90>
     a0e:	00 00 
     a10:	c5 f8 29 44 24 c0    	vmovaps XMMWORD PTR [rsp-0x40],xmm0
     a16:	c4 e2 79 18 05 91 08 	vbroadcastss xmm0,DWORD PTR [rip+0x891]        # 12b0 <_fini+0x94>
     a1d:	00 00 
     a1f:	c5 f8 29 44 24 b0    	vmovaps XMMWORD PTR [rsp-0x50],xmm0
     a25:	c4 e2 79 18 05 86 08 	vbroadcastss xmm0,DWORD PTR [rip+0x886]        # 12b4 <_fini+0x98>
     a2c:	00 00 
     a2e:	c5 f8 29 44 24 a0    	vmovaps XMMWORD PTR [rsp-0x60],xmm0
     a34:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
     a3b:	00 00 00 
     a3e:	66 90                	xchg   ax,ax
     a40:	c4 c1 7a 10 84 91 00 	vmovss xmm0,DWORD PTR [r9+rdx*4+0x3000000]
     a47:	00 00 03 
     a4a:	c5 fa 11 44 24 80    	vmovss DWORD PTR [rsp-0x80],xmm0
     a50:	c5 fa 10 05 d8 07 00 	vmovss xmm0,DWORD PTR [rip+0x7d8]        # 1230 <_fini+0x14>
     a57:	00 
     a58:	c4 c1 7a 59 8c 91 00 	vmulss xmm1,xmm0,DWORD PTR [r9+rdx*4+0x2000000]
     a5f:	00 00 02 
     a62:	c4 c1 7a 59 94 91 00 	vmulss xmm2,xmm0,DWORD PTR [r9+rdx*4+0x7000000]
     a69:	00 00 07 
     a6c:	c4 c1 7a 59 ac 91 00 	vmulss xmm5,xmm0,DWORD PTR [r9+rdx*4+0xc000000]
     a73:	00 00 0c 
     a76:	c5 fa 10 05 b6 07 00 	vmovss xmm0,DWORD PTR [rip+0x7b6]        # 1234 <_fini+0x18>
     a7d:	00 
     a7e:	c5 f2 58 d8          	vaddss xmm3,xmm1,xmm0
     a82:	c5 ea 58 e0          	vaddss xmm4,xmm2,xmm0
     a86:	c5 d2 58 f0          	vaddss xmm6,xmm5,xmm0
     a8a:	c5 78 28 e8          	vmovaps xmm13,xmm0
     a8e:	c5 fa 2c f3          	vcvttss2si esi,xmm3
     a92:	c5 fa 2c fc          	vcvttss2si edi,xmm4
     a96:	c5 7a 2c c6          	vcvttss2si r8d,xmm6
     a9a:	c4 c1 7a 10 84 91 00 	vmovss xmm0,DWORD PTR [r9+rdx*4+0x8000000]
     aa1:	00 00 08 
     aa4:	c5 fa 11 44 24 88    	vmovss DWORD PTR [rsp-0x78],xmm0
     aaa:	c4 c1 7a 10 84 91 00 	vmovss xmm0,DWORD PTR [r9+rdx*4+0xd000000]
     ab1:	00 00 0d 
     ab4:	c5 fa 11 44 24 8c    	vmovss DWORD PTR [rsp-0x74],xmm0
     aba:	c5 c2 2a f6          	vcvtsi2ss xmm6,xmm7,esi
     abe:	c5 c2 2a ff          	vcvtsi2ss xmm7,xmm7,edi
     ac2:	c4 c1 3a 2a c0       	vcvtsi2ss xmm0,xmm8,r8d
     ac7:	c5 72 5c c6          	vsubss xmm8,xmm1,xmm6
     acb:	c5 6a 5c df          	vsubss xmm11,xmm2,xmm7
     acf:	c5 d2 5c d0          	vsubss xmm2,xmm5,xmm0
     ad3:	c5 fa 10 05 5d 07 00 	vmovss xmm0,DWORD PTR [rip+0x75d]        # 1238 <_fini+0x1c>
     ada:	00 
     adb:	c5 f8 28 d8          	vmovaps xmm3,xmm0
     adf:	c4 c1 7a 5c c0       	vsubss xmm0,xmm0,xmm8
     ae4:	c5 78 29 44 24 90    	vmovaps XMMWORD PTR [rsp-0x70],xmm8
     aea:	c4 c1 62 5c cb       	vsubss xmm1,xmm3,xmm11
     aef:	c5 e2 5c ea          	vsubss xmm5,xmm3,xmm2
     af3:	c5 fa 59 c0          	vmulss xmm0,xmm0,xmm0
     af7:	c5 f2 59 f1          	vmulss xmm6,xmm1,xmm1
     afb:	c5 d2 59 ed          	vmulss xmm5,xmm5,xmm5
     aff:	c5 fa 10 0d 35 07 00 	vmovss xmm1,DWORD PTR [rip+0x735]        # 123c <_fini+0x20>
     b06:	00 
     b07:	c5 7a 59 d1          	vmulss xmm10,xmm0,xmm1
     b0b:	c5 4a 59 c9          	vmulss xmm9,xmm6,xmm1
     b0f:	c5 7a 11 4c 24 84    	vmovss DWORD PTR [rsp-0x7c],xmm9
     b15:	c5 52 59 f1          	vmulss xmm14,xmm5,xmm1
     b19:	c5 fa 10 05 1f 07 00 	vmovss xmm0,DWORD PTR [rip+0x71f]        # 1240 <_fini+0x24>
     b20:	00 
     b21:	c5 6a 58 e0          	vaddss xmm12,xmm2,xmm0
     b25:	c5 f8 28 d8          	vmovaps xmm3,xmm0
     b29:	c5 fa 10 05 13 07 00 	vmovss xmm0,DWORD PTR [rip+0x713]        # 1244 <_fini+0x28>
     b30:	00 
     b31:	c4 62 19 ad e0       	vfnmadd213ss xmm12,xmm12,xmm0
     b36:	c4 c1 6a 58 c5       	vaddss xmm0,xmm2,xmm13
     b3b:	c5 fa 59 c0          	vmulss xmm0,xmm0,xmm0
     b3f:	c5 7a 59 e9          	vmulss xmm13,xmm0,xmm1
     b43:	c4 c1 62 58 84 91 00 	vaddss xmm0,xmm3,DWORD PTR [r9+rdx*4+0x1000000]
     b4a:	00 00 01 
     b4d:	c5 fa 59 35 f3 06 00 	vmulss xmm6,xmm0,DWORD PTR [rip+0x6f3]        # 1248 <_fini+0x2c>
     b54:	00 
     b55:	c4 c1 7a 10 84 91 00 	vmovss xmm0,DWORD PTR [r9+rdx*4+0x5000000]
     b5c:	00 00 05 
     b5f:	c5 fa 58 c0          	vaddss xmm0,xmm0,xmm0
     b63:	c4 c1 7a 10 ac 91 00 	vmovss xmm5,DWORD PTR [r9+rdx*4+0x6000000]
     b6a:	00 00 06 
     b6d:	c4 c1 7a 10 9c 91 00 	vmovss xmm3,DWORD PTR [r9+rdx*4+0x9000000]
     b74:	00 00 09 
     b77:	c5 e2 58 db          	vaddss xmm3,xmm3,xmm3
     b7b:	c4 c3 51 21 ac 91 00 	vinsertps xmm5,xmm5,DWORD PTR [r9+rdx*4+0xb000000],0x10
     b82:	00 00 0b 10 
     b86:	c5 fa 10 0d be 06 00 	vmovss xmm1,DWORD PTR [rip+0x6be]        # 124c <_fini+0x30>
     b8d:	00 
     b8e:	c4 e2 79 18 e1       	vbroadcastss xmm4,xmm1
     b93:	c5 50 59 fc          	vmulps xmm15,xmm5,xmm4
     b97:	c4 c1 7a 10 ac 91 00 	vmovss xmm5,DWORD PTR [r9+rdx*4+0x4000000]
     b9e:	00 00 04 
     ba1:	c4 e2 49 99 e9       	vfmadd132ss xmm5,xmm6,xmm1
     ba6:	c5 fa 10 3d a2 06 00 	vmovss xmm7,DWORD PTR [rip+0x6a2]        # 1250 <_fini+0x34>
     bad:	00 
     bae:	c5 f8 28 e7          	vmovaps xmm4,xmm7
     bb2:	c5 d2 59 ef          	vmulss xmm5,xmm5,xmm7
     bb6:	c5 e2 59 df          	vmulss xmm3,xmm3,xmm7
     bba:	c4 e3 51 21 db 10    	vinsertps xmm3,xmm5,xmm3,0x10
     bc0:	c4 c1 7a 10 ac 91 00 	vmovss xmm5,DWORD PTR [r9+rdx*4+0xa000000]
     bc7:	00 00 0a 
     bca:	c4 e2 49 99 e9       	vfmadd132ss xmm5,xmm6,xmm1
     bcf:	c5 fa 59 c7          	vmulss xmm0,xmm0,xmm7
     bd3:	c5 d2 59 ef          	vmulss xmm5,xmm5,xmm7
     bd7:	c4 e3 79 21 c5 10    	vinsertps xmm0,xmm0,xmm5,0x10
     bdd:	c4 e2 79 18 ef       	vbroadcastss xmm5,xmm7
     be2:	c5 00 59 fd          	vmulps xmm15,xmm15,xmm5
     be6:	c4 c1 7a 10 ac 91 00 	vmovss xmm5,DWORD PTR [r9+rdx*4+0xe000000]
     bed:	00 00 0e 
     bf0:	c5 d2 58 ed          	vaddss xmm5,xmm5,xmm5
     bf4:	c5 d2 59 ef          	vmulss xmm5,xmm5,xmm7
     bf8:	c4 e3 61 21 ed 28    	vinsertps xmm5,xmm3,xmm5,0x28
     bfe:	c5 f8 29 ac 24 40 01 	vmovaps XMMWORD PTR [rsp+0x140],xmm5
     c05:	00 00 
     c07:	c4 c1 7a 10 9c 91 00 	vmovss xmm3,DWORD PTR [r9+rdx*4+0xf000000]
     c0e:	00 00 0f 
     c11:	c5 e2 58 db          	vaddss xmm3,xmm3,xmm3
     c15:	c4 c2 71 b9 b4 91 00 	vfmadd231ss xmm6,xmm1,DWORD PTR [r9+rdx*4+0x10000000]
     c1c:	00 00 10 
     c1f:	c5 e2 59 df          	vmulss xmm3,xmm3,xmm7
     c23:	c4 e3 79 21 fb 28    	vinsertps xmm7,xmm0,xmm3,0x28
     c29:	c5 ca 59 c4          	vmulss xmm0,xmm6,xmm4
     c2d:	c4 63 01 21 f8 28    	vinsertps xmm15,xmm15,xmm0,0x28
     c33:	6b c6 31             	imul   eax,esi,0x31
     c36:	8d 34 fd 00 00 00 00 	lea    esi,[rdi*8+0x0]
     c3d:	29 fe                	sub    esi,edi
     c3f:	01 c6                	add    esi,eax
     c41:	44 01 c6             	add    esi,r8d
     c44:	c5 fa 10 44 24 80    	vmovss xmm0,DWORD PTR [rsp-0x80]
     c4a:	c5 fa 58 c0          	vaddss xmm0,xmm0,xmm0
     c4e:	c5 fa 10 5c 24 88    	vmovss xmm3,DWORD PTR [rsp-0x78]
     c54:	c5 e2 58 db          	vaddss xmm3,xmm3,xmm3
     c58:	c4 e3 79 21 c3 10    	vinsertps xmm0,xmm0,xmm3,0x10
     c5e:	c5 fa 10 5c 24 8c    	vmovss xmm3,DWORD PTR [rsp-0x74]
     c64:	c5 e2 58 db          	vaddss xmm3,xmm3,xmm3
     c68:	c4 e3 79 21 c3 20    	vinsertps xmm0,xmm0,xmm3,0x20
     c6e:	c4 c2 79 18 d8       	vbroadcastss xmm3,xmm8
     c73:	c5 e0 59 dd          	vmulps xmm3,xmm3,xmm5
     c77:	c4 c2 79 18 e3       	vbroadcastss xmm4,xmm11
     c7c:	c4 e2 41 a8 e3       	vfmadd213ps xmm4,xmm7,xmm3
     c81:	c4 e2 79 18 d2       	vbroadcastss xmm2,xmm2
     c86:	c4 e2 01 a8 d4       	vfmadd213ps xmm2,xmm15,xmm4
     c8b:	c4 e3 79 21 c1 30    	vinsertps xmm0,xmm0,xmm1,0x30
     c91:	c5 f8 5c da          	vsubps xmm3,xmm0,xmm2
     c95:	c5 78 29 d1          	vmovaps xmm1,xmm10
     c99:	c4 c1 2a 59 e1       	vmulss xmm4,xmm10,xmm9
     c9e:	c5 00 59 0d 1a 06 00 	vmulps xmm9,xmm15,XMMWORD PTR [rip+0x61a]        # 12c0 <_fini+0xa4>
     ca5:	00 
     ca6:	c5 b0 58 d3          	vaddps xmm2,xmm9,xmm3
     caa:	c4 c1 5a 59 f6       	vmulss xmm6,xmm4,xmm14
     caf:	c4 e2 79 18 f6       	vbroadcastss xmm6,xmm6
     cb4:	48 89 f0             	mov    rax,rsi
     cb7:	48 c1 e0 20          	shl    rax,0x20
     cbb:	48 09 f0             	or     rax,rsi
     cbe:	c4 e1 f9 6e c0       	vmovq  xmm0,rax
     cc3:	48 63 c6             	movsxd rax,esi
     cc6:	48 c1 e0 04          	shl    rax,0x4
     cca:	c4 e2 69 a8 34 01    	vfmadd213ps xmm6,xmm2,XMMWORD PTR [rcx+rax*1]
     cd0:	c4 62 79 59 d0       	vpbroadcastq xmm10,xmm0
     cd5:	c5 f8 11 34 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm6
     cda:	c5 80 58 c3          	vaddps xmm0,xmm15,xmm3
     cde:	c5 a9 fa 15 ea 05 00 	vpsubd xmm2,xmm10,XMMWORD PTR [rip+0x5ea]        # 12d0 <_fini+0xb4>
     ce5:	00 
     ce6:	c4 e1 f9 7e d0       	vmovq  rax,xmm2
     ceb:	c4 c1 5a 59 d4       	vmulss xmm2,xmm4,xmm12
     cf0:	c4 e2 79 18 d2       	vbroadcastss xmm2,xmm2
     cf5:	48 98                	cdqe   
     cf7:	48 c1 e0 04          	shl    rax,0x4
     cfb:	c4 e2 79 a8 14 01    	vfmadd213ps xmm2,xmm0,XMMWORD PTR [rcx+rax*1]
     d01:	c5 f8 11 14 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm2
     d06:	c4 c1 00 58 d7       	vaddps xmm2,xmm15,xmm15
     d0b:	c5 a9 fe 84 24 20 01 	vpaddd xmm0,xmm10,XMMWORD PTR [rsp+0x120]
     d12:	00 00 
     d14:	c4 e1 f9 7e c0       	vmovq  rax,xmm0
     d19:	c5 e8 58 c3          	vaddps xmm0,xmm2,xmm3
     d1d:	c4 c1 5a 59 e5       	vmulss xmm4,xmm4,xmm13
     d22:	c4 e2 79 18 f4       	vbroadcastss xmm6,xmm4
     d27:	48 98                	cdqe   
     d29:	48 c1 e0 04          	shl    rax,0x4
     d2d:	c4 e2 79 a8 34 01    	vfmadd213ps xmm6,xmm0,XMMWORD PTR [rcx+rax*1]
     d33:	c5 22 58 05 05 05 00 	vaddss xmm8,xmm11,DWORD PTR [rip+0x505]        # 1240 <_fini+0x24>
     d3a:	00 
     d3b:	c4 62 39 ad 05 00 05 	vfnmadd213ss xmm8,xmm8,DWORD PTR [rip+0x500]        # 1244 <_fini+0x28>
     d42:	00 00 
     d44:	c5 f8 11 34 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm6
     d49:	c5 a9 fe 84 24 10 01 	vpaddd xmm0,xmm10,XMMWORD PTR [rsp+0x110]
     d50:	00 00 
     d52:	c4 e1 f9 7e c0       	vmovq  rax,xmm0
     d57:	c4 c1 72 59 c0       	vmulss xmm0,xmm1,xmm8
     d5c:	c4 c1 7a 59 f6       	vmulss xmm6,xmm0,xmm14
     d61:	c4 e2 79 18 e6       	vbroadcastss xmm4,xmm6
     d66:	48 98                	cdqe   
     d68:	48 c1 e0 04          	shl    rax,0x4
     d6c:	c5 c0 58 f3          	vaddps xmm6,xmm7,xmm3
     d70:	c5 b0 58 ee          	vaddps xmm5,xmm9,xmm6
     d74:	c4 e2 51 a8 24 01    	vfmadd213ps xmm4,xmm5,XMMWORD PTR [rcx+rax*1]
     d7a:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
     d7f:	c5 a9 fe a4 24 00 01 	vpaddd xmm4,xmm10,XMMWORD PTR [rsp+0x100]
     d86:	00 00 
     d88:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
     d8d:	c4 c1 7a 59 e4       	vmulss xmm4,xmm0,xmm12
     d92:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
     d97:	48 98                	cdqe   
     d99:	48 c1 e0 04          	shl    rax,0x4
     d9d:	c5 80 58 ee          	vaddps xmm5,xmm15,xmm6
     da1:	c4 e2 51 a8 24 01    	vfmadd213ps xmm4,xmm5,XMMWORD PTR [rcx+rax*1]
     da7:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
     dac:	c5 a9 fe a4 24 f0 00 	vpaddd xmm4,xmm10,XMMWORD PTR [rsp+0xf0]
     db3:	00 00 
     db5:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
     dba:	c4 c1 7a 59 c5       	vmulss xmm0,xmm0,xmm13
     dbf:	c4 e2 79 18 c0       	vbroadcastss xmm0,xmm0
     dc4:	48 98                	cdqe   
     dc6:	48 c1 e0 04          	shl    rax,0x4
     dca:	c5 e8 58 e6          	vaddps xmm4,xmm2,xmm6
     dce:	c4 e2 59 a8 04 01    	vfmadd213ps xmm0,xmm4,XMMWORD PTR [rcx+rax*1]
     dd4:	c5 f8 11 04 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm0
     dd9:	c5 a9 fe 84 24 e0 00 	vpaddd xmm0,xmm10,XMMWORD PTR [rsp+0xe0]
     de0:	00 00 
     de2:	c4 e1 f9 7e c0       	vmovq  rax,xmm0
     de7:	c5 a2 58 05 45 04 00 	vaddss xmm0,xmm11,DWORD PTR [rip+0x445]        # 1234 <_fini+0x18>
     dee:	00 
     def:	c5 fa 59 c0          	vmulss xmm0,xmm0,xmm0
     df3:	c5 fa 59 25 41 04 00 	vmulss xmm4,xmm0,DWORD PTR [rip+0x441]        # 123c <_fini+0x20>
     dfa:	00 
     dfb:	c5 fa 11 64 24 80    	vmovss DWORD PTR [rsp-0x80],xmm4
     e01:	c5 c0 58 c6          	vaddps xmm0,xmm7,xmm6
     e05:	c5 f2 59 cc          	vmulss xmm1,xmm1,xmm4
     e09:	c4 c1 72 59 e6       	vmulss xmm4,xmm1,xmm14
     e0e:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
     e13:	48 98                	cdqe   
     e15:	48 c1 e0 04          	shl    rax,0x4
     e19:	c5 b0 58 e8          	vaddps xmm5,xmm9,xmm0
     e1d:	c4 e2 51 a8 24 01    	vfmadd213ps xmm4,xmm5,XMMWORD PTR [rcx+rax*1]
     e23:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
     e28:	c5 a9 fe a4 24 d0 00 	vpaddd xmm4,xmm10,XMMWORD PTR [rsp+0xd0]
     e2f:	00 00 
     e31:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
     e36:	c4 c1 72 59 e4       	vmulss xmm4,xmm1,xmm12
     e3b:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
     e40:	48 98                	cdqe   
     e42:	48 c1 e0 04          	shl    rax,0x4
     e46:	c5 80 58 e8          	vaddps xmm5,xmm15,xmm0
     e4a:	c4 e2 51 a8 24 01    	vfmadd213ps xmm4,xmm5,XMMWORD PTR [rcx+rax*1]
     e50:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
     e55:	c5 a9 fe a4 24 c0 00 	vpaddd xmm4,xmm10,XMMWORD PTR [rsp+0xc0]
     e5c:	00 00 
     e5e:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
     e63:	c5 e8 58 c0          	vaddps xmm0,xmm2,xmm0
     e67:	c4 c1 72 59 cd       	vmulss xmm1,xmm1,xmm13
     e6c:	c4 e2 79 18 c9       	vbroadcastss xmm1,xmm1
     e71:	48 98                	cdqe   
     e73:	48 c1 e0 04          	shl    rax,0x4
     e77:	c4 e2 79 a8 0c 01    	vfmadd213ps xmm1,xmm0,XMMWORD PTR [rcx+rax*1]
     e7d:	c5 fa 10 05 bb 03 00 	vmovss xmm0,DWORD PTR [rip+0x3bb]        # 1240 <_fini+0x24>
     e84:	00 
     e85:	c5 fa 58 74 24 90    	vaddss xmm6,xmm0,DWORD PTR [rsp-0x70]
     e8b:	c4 e2 49 ad 35 b0 03 	vfnmadd213ss xmm6,xmm6,DWORD PTR [rip+0x3b0]        # 1244 <_fini+0x28>
     e92:	00 00 
     e94:	c5 f8 11 0c 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm1
     e99:	c5 78 28 9c 24 40 01 	vmovaps xmm11,XMMWORD PTR [rsp+0x140]
     ea0:	00 00 
     ea2:	c5 a0 58 cb          	vaddps xmm1,xmm11,xmm3
     ea6:	c5 a9 fe 84 24 b0 00 	vpaddd xmm0,xmm10,XMMWORD PTR [rsp+0xb0]
     ead:	00 00 
     eaf:	c4 e1 f9 7e c0       	vmovq  rax,xmm0
     eb4:	c5 ca 59 44 24 84    	vmulss xmm0,xmm6,DWORD PTR [rsp-0x7c]
     eba:	c4 c1 7a 59 de       	vmulss xmm3,xmm0,xmm14
     ebf:	c4 e2 79 18 db       	vbroadcastss xmm3,xmm3
     ec4:	48 98                	cdqe   
     ec6:	48 c1 e0 04          	shl    rax,0x4
     eca:	c5 b0 58 e1          	vaddps xmm4,xmm9,xmm1
     ece:	c4 e2 59 a8 1c 01    	vfmadd213ps xmm3,xmm4,XMMWORD PTR [rcx+rax*1]
     ed4:	c5 f8 11 1c 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm3
     ed9:	c5 a9 fe 9c 24 a0 00 	vpaddd xmm3,xmm10,XMMWORD PTR [rsp+0xa0]
     ee0:	00 00 
     ee2:	c4 e1 f9 7e d8       	vmovq  rax,xmm3
     ee7:	c4 c1 7a 59 dc       	vmulss xmm3,xmm0,xmm12
     eec:	c4 e2 79 18 db       	vbroadcastss xmm3,xmm3
     ef1:	48 98                	cdqe   
     ef3:	48 c1 e0 04          	shl    rax,0x4
     ef7:	c5 80 58 e1          	vaddps xmm4,xmm15,xmm1
     efb:	c4 e2 59 a8 1c 01    	vfmadd213ps xmm3,xmm4,XMMWORD PTR [rcx+rax*1]
     f01:	c5 f8 11 1c 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm3
     f06:	c5 a9 fe 9c 24 90 00 	vpaddd xmm3,xmm10,XMMWORD PTR [rsp+0x90]
     f0d:	00 00 
     f0f:	c4 e1 f9 7e d8       	vmovq  rax,xmm3
     f14:	c4 c1 7a 59 c5       	vmulss xmm0,xmm0,xmm13
     f19:	c4 e2 79 18 c0       	vbroadcastss xmm0,xmm0
     f1e:	48 98                	cdqe   
     f20:	48 c1 e0 04          	shl    rax,0x4
     f24:	c5 e8 58 d9          	vaddps xmm3,xmm2,xmm1
     f28:	c4 e2 61 a8 04 01    	vfmadd213ps xmm0,xmm3,XMMWORD PTR [rcx+rax*1]
     f2e:	c5 f8 11 04 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm0
     f33:	c5 a9 fe 84 24 80 00 	vpaddd xmm0,xmm10,XMMWORD PTR [rsp+0x80]
     f3a:	00 00 
     f3c:	c4 e1 f9 7e c0       	vmovq  rax,xmm0
     f41:	c4 c1 4a 59 c0       	vmulss xmm0,xmm6,xmm8
     f46:	c4 c1 7a 59 de       	vmulss xmm3,xmm0,xmm14
     f4b:	c4 e2 79 18 e3       	vbroadcastss xmm4,xmm3
     f50:	48 98                	cdqe   
     f52:	48 c1 e0 04          	shl    rax,0x4
     f56:	c5 f8 29 bc 24 30 01 	vmovaps XMMWORD PTR [rsp+0x130],xmm7
     f5d:	00 00 
     f5f:	c5 c0 58 d9          	vaddps xmm3,xmm7,xmm1
     f63:	c5 b0 58 eb          	vaddps xmm5,xmm9,xmm3
     f67:	c4 e2 51 a8 24 01    	vfmadd213ps xmm4,xmm5,XMMWORD PTR [rcx+rax*1]
     f6d:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
     f72:	c5 a9 fe 64 24 70    	vpaddd xmm4,xmm10,XMMWORD PTR [rsp+0x70]
     f78:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
     f7d:	c4 c1 7a 59 e4       	vmulss xmm4,xmm0,xmm12
     f82:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
     f87:	48 98                	cdqe   
     f89:	48 c1 e0 04          	shl    rax,0x4
     f8d:	c5 80 58 eb          	vaddps xmm5,xmm15,xmm3
     f91:	c4 e2 51 a8 24 01    	vfmadd213ps xmm4,xmm5,XMMWORD PTR [rcx+rax*1]
     f97:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
     f9c:	c5 a9 fe 64 24 60    	vpaddd xmm4,xmm10,XMMWORD PTR [rsp+0x60]
     fa2:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
     fa7:	c4 c1 7a 59 c5       	vmulss xmm0,xmm0,xmm13
     fac:	c4 e2 79 18 c0       	vbroadcastss xmm0,xmm0
     fb1:	48 98                	cdqe   
     fb3:	48 c1 e0 04          	shl    rax,0x4
     fb7:	c5 e8 58 e3          	vaddps xmm4,xmm2,xmm3
     fbb:	c4 e2 59 a8 04 01    	vfmadd213ps xmm0,xmm4,XMMWORD PTR [rcx+rax*1]
     fc1:	c5 f8 11 04 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm0
     fc6:	c5 a9 fe 44 24 50    	vpaddd xmm0,xmm10,XMMWORD PTR [rsp+0x50]
     fcc:	c4 e1 f9 7e c0       	vmovq  rax,xmm0
     fd1:	c5 c0 58 c3          	vaddps xmm0,xmm7,xmm3
     fd5:	c5 fa 10 7c 24 80    	vmovss xmm7,DWORD PTR [rsp-0x80]
     fdb:	c5 ca 59 df          	vmulss xmm3,xmm6,xmm7
     fdf:	c4 c1 62 59 e6       	vmulss xmm4,xmm3,xmm14
     fe4:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
     fe9:	48 98                	cdqe   
     feb:	48 c1 e0 04          	shl    rax,0x4
     fef:	c5 b0 58 e8          	vaddps xmm5,xmm9,xmm0
     ff3:	c4 e2 51 a8 24 01    	vfmadd213ps xmm4,xmm5,XMMWORD PTR [rcx+rax*1]
     ff9:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
     ffe:	c5 a9 fe 64 24 40    	vpaddd xmm4,xmm10,XMMWORD PTR [rsp+0x40]
    1004:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
    1009:	c4 c1 62 59 e4       	vmulss xmm4,xmm3,xmm12
    100e:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
    1013:	48 98                	cdqe   
    1015:	48 c1 e0 04          	shl    rax,0x4
    1019:	c5 80 58 e8          	vaddps xmm5,xmm15,xmm0
    101d:	c4 e2 51 a8 24 01    	vfmadd213ps xmm4,xmm5,XMMWORD PTR [rcx+rax*1]
    1023:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
    1028:	c5 a9 fe 64 24 30    	vpaddd xmm4,xmm10,XMMWORD PTR [rsp+0x30]
    102e:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
    1033:	c5 e8 58 c0          	vaddps xmm0,xmm2,xmm0
    1037:	c4 c1 62 59 dd       	vmulss xmm3,xmm3,xmm13
    103c:	c4 e2 79 18 e3       	vbroadcastss xmm4,xmm3
    1041:	48 98                	cdqe   
    1043:	48 c1 e0 04          	shl    rax,0x4
    1047:	c4 e2 79 a8 24 01    	vfmadd213ps xmm4,xmm0,XMMWORD PTR [rcx+rax*1]
    104d:	c5 f8 28 44 24 90    	vmovaps xmm0,XMMWORD PTR [rsp-0x70]
    1053:	c5 fa 58 05 d9 01 00 	vaddss xmm0,xmm0,DWORD PTR [rip+0x1d9]        # 1234 <_fini+0x18>
    105a:	00 
    105b:	c5 fa 59 c0          	vmulss xmm0,xmm0,xmm0
    105f:	c5 fa 59 1d d5 01 00 	vmulss xmm3,xmm0,DWORD PTR [rip+0x1d5]        # 123c <_fini+0x20>
    1066:	00 
    1067:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
    106c:	c5 a0 58 c9          	vaddps xmm1,xmm11,xmm1
    1070:	c5 e2 59 44 24 84    	vmulss xmm0,xmm3,DWORD PTR [rsp-0x7c]
    1076:	c5 b0 58 e1          	vaddps xmm4,xmm9,xmm1
    107a:	c5 a9 fe 6c 24 20    	vpaddd xmm5,xmm10,XMMWORD PTR [rsp+0x20]
    1080:	c4 e1 f9 7e e8       	vmovq  rax,xmm5
    1085:	c4 c1 7a 59 ee       	vmulss xmm5,xmm0,xmm14
    108a:	c4 e2 79 18 ed       	vbroadcastss xmm5,xmm5
    108f:	48 98                	cdqe   
    1091:	48 c1 e0 04          	shl    rax,0x4
    1095:	c4 e2 59 a8 2c 01    	vfmadd213ps xmm5,xmm4,XMMWORD PTR [rcx+rax*1]
    109b:	c5 f8 11 2c 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm5
    10a0:	c5 80 58 e1          	vaddps xmm4,xmm15,xmm1
    10a4:	c5 a9 fe 6c 24 10    	vpaddd xmm5,xmm10,XMMWORD PTR [rsp+0x10]
    10aa:	c4 e1 f9 7e e8       	vmovq  rax,xmm5
    10af:	c4 c1 7a 59 ec       	vmulss xmm5,xmm0,xmm12
    10b4:	c4 e2 79 18 ed       	vbroadcastss xmm5,xmm5
    10b9:	48 98                	cdqe   
    10bb:	48 c1 e0 04          	shl    rax,0x4
    10bf:	c4 e2 59 a8 2c 01    	vfmadd213ps xmm5,xmm4,XMMWORD PTR [rcx+rax*1]
    10c5:	c5 f8 11 2c 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm5
    10ca:	c5 a9 fe 24 24       	vpaddd xmm4,xmm10,XMMWORD PTR [rsp]
    10cf:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
    10d4:	c5 e8 58 e1          	vaddps xmm4,xmm2,xmm1
    10d8:	c4 c1 7a 59 c5       	vmulss xmm0,xmm0,xmm13
    10dd:	c4 e2 79 18 c0       	vbroadcastss xmm0,xmm0
    10e2:	48 98                	cdqe   
    10e4:	48 c1 e0 04          	shl    rax,0x4
    10e8:	c4 e2 59 a8 04 01    	vfmadd213ps xmm0,xmm4,XMMWORD PTR [rcx+rax*1]
    10ee:	c5 f8 11 04 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm0
    10f3:	c5 f8 28 b4 24 30 01 	vmovaps xmm6,XMMWORD PTR [rsp+0x130]
    10fa:	00 00 
    10fc:	c5 c8 58 c9          	vaddps xmm1,xmm6,xmm1
    1100:	c4 c1 62 59 c0       	vmulss xmm0,xmm3,xmm8
    1105:	c5 b0 58 e1          	vaddps xmm4,xmm9,xmm1
    1109:	c5 a9 fe 6c 24 f0    	vpaddd xmm5,xmm10,XMMWORD PTR [rsp-0x10]
    110f:	c4 e1 f9 7e e8       	vmovq  rax,xmm5
    1114:	c4 c1 7a 59 ee       	vmulss xmm5,xmm0,xmm14
    1119:	c4 e2 79 18 ed       	vbroadcastss xmm5,xmm5
    111e:	48 98                	cdqe   
    1120:	48 c1 e0 04          	shl    rax,0x4
    1124:	c4 e2 59 a8 2c 01    	vfmadd213ps xmm5,xmm4,XMMWORD PTR [rcx+rax*1]
    112a:	c5 f8 11 2c 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm5
    112f:	c5 80 58 e1          	vaddps xmm4,xmm15,xmm1
    1133:	c5 a9 fe 6c 24 e0    	vpaddd xmm5,xmm10,XMMWORD PTR [rsp-0x20]
    1139:	c4 e1 f9 7e e8       	vmovq  rax,xmm5
    113e:	c4 c1 7a 59 ec       	vmulss xmm5,xmm0,xmm12
    1143:	c4 e2 79 18 ed       	vbroadcastss xmm5,xmm5
    1148:	48 98                	cdqe   
    114a:	48 c1 e0 04          	shl    rax,0x4
    114e:	c4 e2 59 a8 2c 01    	vfmadd213ps xmm5,xmm4,XMMWORD PTR [rcx+rax*1]
    1154:	c5 f8 11 2c 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm5
    1159:	c5 a9 fe 64 24 d0    	vpaddd xmm4,xmm10,XMMWORD PTR [rsp-0x30]
    115f:	c4 e1 f9 7e e0       	vmovq  rax,xmm4
    1164:	c5 e8 58 e1          	vaddps xmm4,xmm2,xmm1
    1168:	c4 c1 7a 59 c5       	vmulss xmm0,xmm0,xmm13
    116d:	c4 e2 79 18 c0       	vbroadcastss xmm0,xmm0
    1172:	48 98                	cdqe   
    1174:	48 c1 e0 04          	shl    rax,0x4
    1178:	c4 e2 59 a8 04 01    	vfmadd213ps xmm0,xmm4,XMMWORD PTR [rcx+rax*1]
    117e:	c5 f8 11 04 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm0
    1183:	c5 c8 58 c1          	vaddps xmm0,xmm6,xmm1
    1187:	c5 e2 59 cf          	vmulss xmm1,xmm3,xmm7
    118b:	c5 a9 fe 5c 24 c0    	vpaddd xmm3,xmm10,XMMWORD PTR [rsp-0x40]
    1191:	c4 e1 f9 7e d8       	vmovq  rax,xmm3
    1196:	c5 b0 58 d8          	vaddps xmm3,xmm9,xmm0
    119a:	c4 c1 72 59 e6       	vmulss xmm4,xmm1,xmm14
    119f:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
    11a4:	48 98                	cdqe   
    11a6:	48 c1 e0 04          	shl    rax,0x4
    11aa:	c4 e2 61 a8 24 01    	vfmadd213ps xmm4,xmm3,XMMWORD PTR [rcx+rax*1]
    11b0:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
    11b5:	c5 a9 fe 5c 24 b0    	vpaddd xmm3,xmm10,XMMWORD PTR [rsp-0x50]
    11bb:	c4 e1 f9 7e d8       	vmovq  rax,xmm3
    11c0:	c5 80 58 d8          	vaddps xmm3,xmm15,xmm0
    11c4:	c4 c1 72 59 e4       	vmulss xmm4,xmm1,xmm12
    11c9:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
    11ce:	48 98                	cdqe   
    11d0:	48 c1 e0 04          	shl    rax,0x4
    11d4:	c4 e2 61 a8 24 01    	vfmadd213ps xmm4,xmm3,XMMWORD PTR [rcx+rax*1]
    11da:	c5 f8 11 24 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm4
    11df:	c5 e8 58 c0          	vaddps xmm0,xmm2,xmm0
    11e3:	c4 c1 72 59 cd       	vmulss xmm1,xmm1,xmm13
    11e8:	c5 a9 fe 54 24 a0    	vpaddd xmm2,xmm10,XMMWORD PTR [rsp-0x60]
    11ee:	c4 e1 f9 7e d0       	vmovq  rax,xmm2
    11f3:	c4 e2 79 18 c9       	vbroadcastss xmm1,xmm1
    11f8:	48 98                	cdqe   
    11fa:	48 c1 e0 04          	shl    rax,0x4
    11fe:	c4 e2 79 a8 0c 01    	vfmadd213ps xmm1,xmm0,XMMWORD PTR [rcx+rax*1]
    1204:	c5 f8 11 0c 01       	vmovups XMMWORD PTR [rcx+rax*1],xmm1
    1209:	48 83 c2 01          	add    rdx,0x1
    120d:	0f 85 2d f8 ff ff    	jne    a40 <func000001+0x1c0>
    1213:	48 81 c4 58 01 00 00 	add    rsp,0x158
    121a:	c3                   	ret    

Disassembly of section .fini:

000000000000121c <_fini>:
    121c:	48 83 ec 08          	sub    rsp,0x8
    1220:	48 83 c4 08          	add    rsp,0x8
    1224:	c3                   	ret    
